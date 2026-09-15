//! Chrome automation tool.

use crate::chrome::contract::{
    ChromeResponse, ERROR_PAGE_PROBE_JS, EXPECT_TIMEOUT_NOTE, classify_call_failure, eval_count,
    expect_outcome, extract_output, extract_snapshot_text, is_daemon_unavailable_code,
    is_daemon_unavailable_error, is_unreachable_tab_error, net_error_phrase,
    parse_error_page_probe, sanitize_timeout_message, unreachable_tab_message,
    with_condition_timeout_note,
};
use crate::chrome::escape_js_single_quoted;
use crate::chrome::forms::{
    ExpectCond, ExtractGate, count_eval_js, expect_args, extract_gate, parse_count_op,
    parse_predicate, parse_state, validate_extract_getters, wait_args, wait_target,
};
use crate::chrome::spawn::{CliRun, CliSpawn, CliTimeout, spawn_cli};
use crate::util::{UnwrapPoison, is_http_url};
use crate::{Tool, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

/// Actions for navigating and extracting content from web pages.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChromeAction {
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
    /// Useful for submitting forms after filling inputs. `selector` tries to
    /// focus the element before pressing, but focus moves only if it is
    /// focusable (input, textarea, button, ...); for a non-focusable selector
    /// (e.g. `body`) the key lands wherever focus currently is.
    Press {
        key: String,
        #[serde(default)]
        selector: Option<String>,
    },
    /// Clear an input/textarea/contenteditable and fill it with text — the
    /// written value is read back and verified, so success means the field
    /// really holds it. Works on rich editors and framework inputs natively.
    Fill { selector: String, text: String },
    /// Type text character-by-character, appending without clearing.
    /// Embedded newlines press Enter — use fill for multiline text.
    Type {
        selector: String,
        text: String,
        #[serde(default)]
        key_events: bool,
    },
    /// Run JavaScript in the page context. Returns the result as a string.
    /// Useful for inspecting element attributes, checking state, or debugging.
    Eval { js: String },
    /// Wait until a CSS selector matches, the URL matches a pattern, or text
    /// appears — exactly one target. Bounded mahbot-side; the numeric sleep
    /// form does not exist.
    Wait {
        /// CSS selector to wait for.
        #[serde(default)]
        selector: Option<String>,
        /// URL pattern to wait for.
        #[serde(default)]
        url: Option<String>,
        /// Text to wait for in the page.
        #[serde(default)]
        text: Option<String>,
    },
    /// Assert a page condition with a bounded wait — PASS/FAIL output.
    /// Conditions: visible | hidden | present | count | text | value | attr | url.
    Expect {
        condition: String,
        /// CSS selector (required for every condition except url).
        #[serde(default)]
        selector: Option<String>,
        /// Comparison for condition "count": == != > < >= <= (default ==).
        #[serde(default)]
        op: Option<String>,
        /// Expected count (condition "count").
        #[serde(default)]
        count: Option<u64>,
        /// Comparison for text/value/attr/url: equals | contains | matches
        /// (default equals).
        #[serde(default)]
        predicate: Option<String>,
        /// Attribute name (condition "attr").
        #[serde(default)]
        name: Option<String>,
        /// Expected value for text/value/attr; the URL pattern for url.
        #[serde(default)]
        expected: Option<String>,
    },
    /// Schema-driven row extraction. An empty region is reported honestly
    /// (chrome-use's phantom-row auto-detect fallback is gated away), and
    /// `limit` trims rows mahbot-side while `total` keeps the honest count.
    Extract {
        /// Extraction schema (JSON object). Include a "rows" CSS selector key
        /// for row-list extraction, plus one field-per-selector.
        schema: Value,
        /// Trim rows to this many; the output's `total` keeps the honest count.
        #[serde(default)]
        limit: Option<usize>,
    },
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

/// Root session namespace of the interactive chrome tool. Every agent-run session
/// lives under `agent-tab-`: that keeps it out of the daemon orphan sweep, and
/// inside the protected-session set of the `mahbot chrome` frontend, which refuses
/// a bare `session stop` on one without `--force` — so the prefix must not change.
/// An `agent-tab-*` session belongs to the run that opened it: mahbot's own sweeps
/// leave every one of them alone, and that run's release is their closer (the
/// `--force` frontend verb above is the deliberate exception).
const AGENT_TAB_PREFIX: &str = "agent-tab-";

/// Logical name of the per-run default session.
const DEFAULT_TAB: &str = "default";

/// Physical session namespace of one agent run: `agent-tab-<key>-`, where `key`
/// is a hex digest of the run's agent id — the identity the daemon re-reads from
/// durable state when it re-drives an interrupted run. Every segment of one run
/// therefore resolves the SAME physical session names, so a run resumed after a
/// daemon restart or self-update re-attaches to the tabs it was working with
/// (chrome-use re-attaches a session's tabs by name) and its run-end release
/// addresses those same names. 12 hex chars (48 bits) is ample for the distinct
/// agent ids of one machine, and a digest is stable across binary versions — the
/// namespace must survive a self-update.
///
/// One exception to that stability, at the per-tab level: a logical name the
/// sanitizer had to change carries a `DefaultHasher` suffix (`resolve_session`),
/// whose algorithm has no cross-release guarantee — such a tab is not re-attached
/// after a self-update that changed it. Its group is still closed at run end: the
/// name it was opened under is the one the run recorded.
///
/// The price of deriving the name rather than minting it: a session opened under a
/// scheme no agent id derives from is addressable by no release, so nothing can
/// reclaim it — the cost paid by the runs in flight when per-boot random names were
/// replaced.
fn run_session_namespace(agent_id: &str) -> String {
    let hex = crate::util::hex_string(&Sha256::digest(agent_id.as_bytes()));
    format!("{AGENT_TAB_PREFIX}{}-", &hex[..12])
}

/// Every chrome session name ONE agent run's chrome tooling opened, recorded so
/// that run's release can address exactly those names — never an enumeration.
#[derive(Default)]
pub(crate) struct ChromeRunSessions {
    /// Physical namespace of the run this tracker belongs to (empty for the
    /// shared non-agent instance, which passes names through unchanged).
    namespace: String,
    names: std::sync::Mutex<BTreeSet<String>>,
}

impl ChromeRunSessions {
    /// Tracker for one agent run, registered as live for as long as it lives, so the
    /// live-namespace guard keeps an earlier run's release off the names a later
    /// segment of the same id resolves (see
    /// [`crate::tools::chrome_release::register_run_namespace`], and
    /// [`run_session_namespace`] for why the two resolve alike).
    pub(crate) fn for_run(agent_id: &str) -> std::sync::Arc<Self> {
        let namespace = run_session_namespace(agent_id);
        super::chrome_release::register_run_namespace(&namespace);
        std::sync::Arc::new(Self {
            namespace,
            names: std::sync::Mutex::new(BTreeSet::new()),
        })
    }

    /// Physical session namespace of this run — the prefix every logical tab
    /// name of the run is resolved under.
    pub(crate) fn namespace(&self) -> &str {
        &self.namespace
    }

    pub(crate) fn track(&self, name: &str) {
        self.names.lock().unwrap_poison().insert(name.to_string());
    }

    pub(crate) fn snapshot(&self) -> Vec<String> {
        self.names.lock().unwrap_poison().iter().cloned().collect()
    }
}

impl Drop for ChromeRunSessions {
    fn drop(&mut self) {
        super::chrome_release::unregister_run_namespace(&self.namespace);
    }
}

/// Chrome tool for fetching content from web pages.
///
/// Each operation requires a `tab` name — separate chrome sessions
/// (isolated via `--session`). Every session an agent run dispatches is
/// resolved to the run's private namespace (`ChromeTool::resolve_session` plus
/// `ChromeRunSessions`) and handed to that run's release, so concurrent runs of
/// different ids can never collide on or address each other's sessions (see
/// `run_session_namespace` for the same-id case and its transitional residue); tool
/// output keeps showing the logical tab name. Operations on the same tab are
/// serialized via a per-tab lock.
#[derive(Default)]
pub struct ChromeTool {
    /// Per-tab locks — only serializes operations on the same tab.
    /// Different tabs can run concurrently without blocking each other.
    tab_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Path of the most recent screenshot written by a `screenshot` action.
    /// Read by `image_payload` to attach the PNG as a native image part. The
    /// call is guarded by the action being `Screenshot`, so a stale value from
    /// a prior round can never be re-attached to a non-screenshot call.
    last_screenshot: std::sync::Mutex<Option<String>>,
    /// Session namespace prefix for this instance (one agent run's
    /// `agent-tab-<key>-`, empty for the shared non-agent instance — link
    /// enrichment keeps its own sweepable `link-enricher-*` sessions).
    session_prefix: String,
    /// Chrome sessions this run's chrome tooling used — handed to the run's
    /// release when the run ends.
    chrome_sessions: std::sync::Arc<ChromeRunSessions>,
}

impl ChromeTool {
    /// Construct a chrome tool for one agent run, sharing the run's session
    /// tracker: its namespace prefixes every session the run dispatches, and the
    /// run's own names are the ones released at run end.
    pub(crate) fn new(chrome_sessions: std::sync::Arc<ChromeRunSessions>) -> Self {
        Self {
            session_prefix: chrome_sessions.namespace().to_string(),
            chrome_sessions,
            ..Default::default()
        }
    }

    /// Resolve a logical tab name to the physical chrome-use session name:
    /// this run's namespace prefix plus the sanitized tab (the sanitizer
    /// yields exactly chrome-use's allowed charset). Idempotent — an already
    /// prefixed (physical) name passes through unchanged, so an echoed-back
    /// session name can never double-prefix into a different session. The
    /// shared non-agent instance (empty prefix) passes names through
    /// unchanged.
    fn resolve_session(&self, tab: &str) -> String {
        if self.session_prefix.is_empty() || tab.starts_with(&self.session_prefix) {
            return tab.to_string();
        }
        let clean = sanitize_filename_component(tab);
        if clean == tab {
            return format!("{}{clean}", self.session_prefix);
        }
        // The sanitizer collapses distinct names (per-char mapping plus the
        // 40-char truncation); suffixing the raw name's hash keeps distinct
        // logical tabs from collapsing into one physical session. That hasher is
        // `std::hash::DefaultHasher` — deterministic within a build, but with no
        // cross-release guarantee, which is the one place a run's names can change
        // under it (see `run_session_namespace`).
        let mut hasher = DefaultHasher::new();
        tab.hash(&mut hasher);
        format!(
            // Low 32 bits — an 8-hex-char suffix is plenty to disambiguate
            // the handful of tabs one agent run uses.
            "{}{clean}-{:08x}",
            self.session_prefix,
            hasher.finish() & 0xffff_ffff
        )
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
    /// tree returned by the `Open` chrome action (which contains element refs,
    /// ARIA roles, and indentation), this returns plain rendered text — no
    /// markup, no hidden content, no `<script>`/`<style>` noise.
    ///
    /// Falls back to `textContent` if the JavaScript eval fails.
    ///
    /// The tab is left open — an agent run's tab is kept until the run ends (its
    /// release closes it); the per-tab lock is held for the full duration
    /// (navigate + extract) so concurrent callers targeting the same tab are
    /// serialized consistently.
    pub async fn fetch_page_text(&self, url: &str, tab: &str) -> anyhow::Result<String> {
        crate::chrome::validate_url(url)?;

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

    /// Close a chrome session tab by name — verified: the session's tab group
    /// is swept (enumerate → close → re-enumerate convergence) so a leftover
    /// cannot be orphaned silently by a kill-based close. Only the target
    /// session's own tabs are touched, and only the shared instance sweeps at all:
    /// a run's own tooling is handed its LOGICAL tab names, which are not session
    /// names, and its namespaced sessions are the run-end release's business.
    pub async fn close_session(&self, tab: &str) {
        if self.session_prefix.is_empty() {
            super::chrome_daemon::sweep_session(tab).await;
        }
    }

    /// Fail with a cause when the response shows a failed navigation: the tab never
    /// left its `about:blank` scratch (the navigation never committed, so the session
    /// is closed — a no-op for a run's own instance, whose namespaced sessions are the
    /// run-end release's business), or Chrome committed to its error page (the tab stays
    /// open, reusable for a retry); no-op when the navigation committed.
    async fn bail_on_failed_navigation(
        &self,
        tab: &str,
        url: &str,
        response: &ChromeResponse,
    ) -> anyhow::Result<()> {
        let Some(committed_url) = response
            .data
            .as_ref()
            .and_then(|d| d.get("url"))
            .and_then(Value::as_str)
        else {
            return Ok(());
        };
        if crate::chrome::is_chrome_error_page(committed_url) {
            // Best-effort: ask the error page which net error code Chrome
            // rendered (the same probe the CLI `open` uses); the generic
            // message is the fallback when nothing recognizable renders.
            let code = self
                .run_command(&["eval", ERROR_PAGE_PROBE_JS], tab)
                .await
                .ok()
                .and_then(|r| parse_error_page_probe(&r))
                .filter(|p| p.is_error_page)
                .and_then(|p| p.code);
            match code {
                Some(code) => anyhow::bail!(
                    "Navigation to {url} failed — Chrome rendered its error page: {code} ({}). \
                     Verify the URL and network.",
                    net_error_phrase(&code)
                ),
                None => anyhow::bail!(
                    "Navigation to {url} failed — Chrome landed on its error page \
                     (chrome-error://chromewebdata/), meaning the site is unreachable (DNS failure, \
                     refused connection, or a blocked/unsafe port). Verify the URL and network."
                ),
            }
        }
        if crate::chrome::is_blank_page_url(committed_url) {
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
        match super::chrome_daemon::cli_probe().await {
            super::chrome_daemon::CliStatus::Available => {}
            super::chrome_daemon::CliStatus::Missing => {
                anyhow::bail!(
                    "chrome-use CLI is not available. {}",
                    super::chrome_daemon::CHROME_USE_INSTALL_HINT
                );
            }
            super::chrome_daemon::CliStatus::Transient(failure) => {
                let msg = match failure {
                    super::chrome_daemon::CliProbeFailure::Spawn(reason) => format!(
                        "chrome-use CLI check could not spawn the binary ({reason}) — a \
                         temporary failure (e.g. system resource exhaustion), not a missing \
                         install. Retry shortly."
                    ),
                    super::chrome_daemon::CliProbeFailure::BadVersion(status) => format!(
                        "chrome-use CLI is installed but its `--version` check failed \
                         ({status}) — the install looks broken. chrome-use self-updates \
                         once per boot, which only replaces the binary when a newer \
                         release exists; check the logs for the last update error, or \
                         reinstall manually via the chrome-use CLI if you want \
                         hands-on control."
                    ),
                    super::chrome_daemon::CliProbeFailure::Timeout => {
                        "chrome-use CLI probe timed out — the binary is present but \
                         unresponsive. Retry shortly; if this persists the CLI may be wedged."
                            .to_string()
                    }
                };
                anyhow::bail!(msg);
            }
        }
        if !super::chrome_daemon::is_available().await {
            anyhow::bail!("{}", super::chrome_daemon::daemon_down_message());
        }
        Ok(())
    }

    /// Declared chrome-side condition-wait deadlines, forwarded to chrome-use
    /// via `--timeout` (see `build_args`); the mahbot-side kill rides
    /// [`crate::chrome::DEADLINE_SLACK`] above them (see `call_timeout`).
    const WAIT_DEADLINE: Duration = Duration::from_secs(10);
    const EXPECT_DEADLINE: Duration = Duration::from_secs(20);

    /// Mahbot-side per-call kill bounds. chrome-use's own `--timeout` is
    /// ignored by `wait --load networkidle` (which instead uses the seeded
    /// AGENT_BROWSER_DEFAULT_TIMEOUT — 15s; 25s is chrome-use's fallback — so
    /// the mahbot-side bound below always dominates) and a wedged daemon hangs
    /// the CLI in its ~152s retry loop, so the tool
    /// bounds every dispatch itself: open = [`crate::chrome::DEFAULT_OPEN_TIMEOUT`]
    /// plus slack (whole-operation budget), wait/expect = their declared
    /// deadlines plus slack (condition waits), everything else=8s (the
    /// daemon-side `run_cli_bounded` bound).
    fn call_timeout(args: &[&str]) -> Duration {
        use crate::chrome::{DEADLINE_SLACK, DEFAULT_OPEN_TIMEOUT};
        match args.first() {
            Some(&"open") => DEFAULT_OPEN_TIMEOUT + DEADLINE_SLACK,
            Some(&"wait") => Self::WAIT_DEADLINE + DEADLINE_SLACK,
            Some(&"expect") => Self::EXPECT_DEADLINE + DEADLINE_SLACK,
            _ => Duration::from_secs(8),
        }
    }

    /// Run an chrome-use command and parse the JSON response.
    async fn run_command(&self, args: &[&str], tab: &str) -> anyhow::Result<ChromeResponse> {
        let cli = super::chrome_daemon::cli_path().with_context(|| {
            format!(
                "chrome-use CLI is not available. {}",
                super::chrome_daemon::CHROME_USE_INSTALL_HINT
            )
        })?;
        // Resolve the logical tab to this run's namespaced physical session
        // at the single dispatch point, and record the physical name after
        // the CLI path resolved, so a missing CLI never registers a pointless
        // release — the run-end release queue only needs sessions that were
        // actually dispatched.
        let session = self.resolve_session(tab);
        self.chrome_sessions.track(&session);

        let mut logged_args: Vec<&str> = args.to_vec();
        logged_args.extend(["--json", "--session", &session]);
        debug!("chrome-use args: {:?}", logged_args);

        // `open` takes no --timeout flag (chrome-use uses its env default), so
        // its chrome-side deadline is set to the declared budget via the env
        // override; the mahbot kill rides DEADLINE_SLACK above and only fires
        // when the deadline is actually exceeded.
        let bound = Self::call_timeout(args);
        let chrome_deadline =
            (args.first() == Some(&"open")).then_some(crate::chrome::DEFAULT_OPEN_TIMEOUT);

        let run = spawn_cli(CliSpawn {
            path: &cli,
            args,
            session: Some(&session),
            json: true,
            capture_stderr: true,
            timeout: CliTimeout::Bounded(bound),
            // A timed-out/cancelled call must not leave the chrome-use child
            // running its retry loop in the background.
            cancel_kills: true,
            input: None,
            chrome_deadline,
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
                // Probed with the RESOLVED session: that is what the timed-out
                // call actually dispatched against.
                if let Some(down_message) =
                    super::chrome_daemon::health_after_call_timeout(&session).await
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
        let stderr = String::from_utf8_lossy(&output.stderr);
        match parse_run_output(
            args.first().copied(),
            output.status.success(),
            &stdout,
            &stderr,
        ) {
            ParsedRun::Ok(resp) => Ok(resp),
            ParsedRun::Unparseable => {
                anyhow::bail!("Failed to parse chrome-use JSON response")
            }
            ParsedRun::Failed {
                error,
                code,
                retryable,
            } => {
                let error_msg = if error.is_empty() {
                    format!("chrome-use exited with code {}", output.status)
                } else {
                    enhance_chrome_error(error)
                };
                // Classify once: strip chrome-use's stale-relay hint from
                // Timeout messages (mahbot's surface has no `connect` verb;
                // Environment diagnostics pass through untouched), then
                // append the action's remediation note from the same kind.
                let kind = classify_call_failure(code.as_deref(), &error_msg);
                let error_msg = sanitize_timeout_message(kind, &error_msg);
                Self::fail_fast_if_daemon_down(&error_msg, code.as_deref())?;
                let mut error_msg = with_retry_hint(error_msg, retryable);
                with_condition_timeout_note(
                    args.first().copied().unwrap_or(""),
                    kind,
                    &mut error_msg,
                );
                anyhow::bail!("{}: {error_msg}", kind.as_str());
            }
        }
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
        if is_unreachable_tab_error(error) {
            anyhow::bail!("{}", unreachable_tab_message(error));
        }
        if is_daemon_unavailable_error(error) || is_daemon_unavailable_code(code) {
            super::chrome_daemon::note_unhealthy(error);
            anyhow::bail!("{}", super::chrome_daemon::daemon_down_message());
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
    #[expect(clippy::too_many_lines)] // the per-action argv shapes are one cohesive dispatch
    fn build_args(action: &ChromeAction) -> anyhow::Result<Vec<String>> {
        match action {
            ChromeAction::Open { url } => {
                crate::chrome::validate_url(url)?;
                Ok(vec!["open".into(), url.clone()])
            }
            ChromeAction::Snapshot {
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
            ChromeAction::Click { selector } => Ok(vec!["click".into(), selector.clone()]),
            ChromeAction::GetText { selector } => {
                Ok(vec!["get".into(), "text".into(), selector.clone()])
            }
            ChromeAction::GetInnerText { .. } => {
                anyhow::bail!("GetInnerText is handled in execute(), not build_args")
            }
            ChromeAction::GetUrl { .. } => Ok(vec!["get".into(), "url".into()]),
            ChromeAction::Press { key, selector } => {
                let mut args = vec!["press".into(), key.clone()];
                if let Some(sel) = selector {
                    args.extend(["--selector".into(), sel.clone()]);
                }
                Ok(args)
            }
            ChromeAction::Fill { selector, text } => {
                let mut args = vec!["fill".into(), selector.clone()];
                args.extend(crate::chrome::forms::text_value_argv(text));
                Ok(args)
            }
            ChromeAction::Type {
                selector,
                text,
                key_events,
            } => {
                let mut args = vec!["type".into(), selector.clone()];
                if *key_events {
                    // Action flags BEFORE the `--` shield (everything after
                    // it is a verbatim value).
                    args.push("--key-events".into());
                }
                args.extend(crate::chrome::forms::text_value_argv(text));
                Ok(args)
            }
            ChromeAction::Eval { js } => Ok(vec!["eval".into(), js.clone()]),
            ChromeAction::Find {
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
            ChromeAction::Screenshot { .. } => {
                anyhow::bail!("Screenshot is handled in execute(), not build_args")
            }
            ChromeAction::Wait {
                selector,
                url,
                text,
            } => {
                let target = wait_target(selector.as_deref(), url.as_deref(), text.as_deref())
                    .map_err(anyhow::Error::msg)?;
                Ok(wait_args(&target, Self::WAIT_DEADLINE.as_millis()))
            }
            ChromeAction::Expect {
                condition,
                selector,
                op,
                count,
                predicate,
                name,
                expected,
            } => {
                let cond = Self::build_expect_cond(
                    condition,
                    selector.as_ref(),
                    op.as_ref(),
                    count.as_ref(),
                    predicate.as_ref(),
                    name.as_ref(),
                    expected.as_ref(),
                )?;
                Ok(expect_args(&cond, Self::EXPECT_DEADLINE.as_millis()))
            }
            ChromeAction::Extract { .. } => {
                anyhow::bail!("Extract is handled in execute(), not build_args")
            }
        }
    }

    /// Map the model-facing expect fields into a validated [`ExpectCond`].
    /// Validation errors are corrective: they name the allowed values so the
    /// model can self-correct in one round-trip.
    fn build_expect_cond(
        condition: &str,
        selector: Option<&String>,
        op: Option<&String>,
        count: Option<&u64>,
        predicate: Option<&String>,
        name: Option<&String>,
        expected: Option<&String>,
    ) -> anyhow::Result<ExpectCond> {
        let require_selector = |what: &str| {
            selector.cloned().ok_or_else(|| {
                anyhow::anyhow!("expect condition \"{condition}\" requires \"selector\" ({what})")
            })
        };
        // Only the comparison conditions take a predicate — a stray one on a
        // state/count condition is ignored, not an error.
        let require_predicate = || {
            parse_predicate(predicate.map_or("equals", String::as_str)).ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid predicate '{predicate:?}' — use equals | contains | matches"
                )
            })
        };
        Ok(match condition {
            "visible" | "hidden" | "present" => ExpectCond::State {
                selector: require_selector("a CSS selector to check")?,
                state: parse_state(condition).expect("condition matched the state allowlist"),
            },
            "count" => ExpectCond::Count {
                selector: require_selector("the CSS selector to count")?,
                op: parse_count_op(op.map_or("==", String::as_str)).ok_or_else(|| {
                    anyhow::anyhow!("invalid count op '{op:?}' — use == != > < >= <=")
                })?,
                n: count.copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "expect condition \"count\" requires \"count\" (the expected number)"
                    )
                })?,
            },
            "text" | "value" => {
                let predicate = require_predicate()?;
                let selector = require_selector("the element to read")?;
                let value = expected.cloned().ok_or_else(|| {
                    anyhow::anyhow!("expect condition \"{condition}\" requires \"expected\"")
                })?;
                if condition == "text" {
                    ExpectCond::Text {
                        selector,
                        predicate,
                        value,
                    }
                } else {
                    ExpectCond::Value {
                        selector,
                        predicate,
                        value,
                    }
                }
            }
            "attr" => {
                let predicate = require_predicate()?;
                ExpectCond::Attr {
                    selector: require_selector("the element holding the attribute")?,
                    name: name.cloned().ok_or_else(|| {
                        anyhow::anyhow!(
                            "expect condition \"attr\" requires \"name\" (the attribute name)"
                        )
                    })?,
                    predicate,
                    value: expected.cloned().ok_or_else(|| {
                        anyhow::anyhow!("expect condition \"attr\" requires \"expected\"")
                    })?,
                }
            }
            "url" => {
                let predicate = require_predicate()?;
                ExpectCond::Url {
                    predicate,
                    pattern: expected.cloned().ok_or_else(|| {
                        anyhow::anyhow!(
                            "expect condition \"url\" requires \"expected\" (the URL pattern)"
                        )
                    })?,
                }
            }
            other => anyhow::bail!(
                "invalid expect condition '{other}' — use visible | hidden | present | count | text | value | attr | url"
            ),
        })
    }
}

/// Close all running chrome sessions at shutdown, leaving the run-owned
/// `agent-tab-*` ones alone (`AGENT_TAB_PREFIX` owns that rule). Called through
/// `chrome_release::flush_and_close_all_chrome_sessions`, the one chrome shutdown
/// entry point.
///
/// The chrome-use child process does not always get reaped on process exit — its
/// sessions hold open ports and lingering instances that can interfere with the next
/// daemon startup.
pub(crate) async fn close_all_chrome_sessions() {
    if tokio::time::timeout(SHUTDOWN_CLEANUP_TIMEOUT, close_all_chrome_sessions_inner())
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
/// `ChromeResponse::verdict` first — this only extracts the names and returns
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

async fn close_all_chrome_sessions_inner() {
    let Some(cmd) = super::chrome_daemon::cli_path() else {
        tracing::debug!("chrome-use not available, skipping chrome cleanup");
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
        input: None,
        chrome_deadline: None,
    })
    .await
    {
        CliRun::Output(o) => o,
        CliRun::SpawnFailure => {
            tracing::debug!("chrome-use not available, skipping chrome cleanup: spawn failed");
            return;
        }
        CliRun::TimedOut => {
            tracing::debug!("chrome-use session list timed out, skipping chrome cleanup");
            return;
        }
    };

    let mut sessions: Vec<String> = match serde_json::from_slice::<Value>(&list_output.stdout) {
        Ok(v) => {
            // Gate only on an explicit failure verdict; a payload with neither
            // verdict key proceeds (tolerance-first — unknown future envelopes
            // still get their sessions closed if they carry a sessions array).
            if ChromeResponse::from_value(&v).verdict() == Some(false) {
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

    // Agent-run sessions belong to their run and are never closed here
    // (see `chrome_release`).
    let before = sessions.len();
    sessions.retain(|s| !s.starts_with(AGENT_TAB_PREFIX));
    let agent_sessions = before - sessions.len();
    if agent_sessions > 0 {
        tracing::debug!(agent_sessions, "left agent-run chrome session(s) open");
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
                    input: None,
                    chrome_deadline: None,
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
impl Tool for ChromeTool {
    fn name(&self) -> &'static str {
        "chrome"
    }

    fn is_advertised(&self) -> bool {
        super::chrome_daemon::is_advertised()
    }

    fn parameters_schema(&self) -> Value {
        // oneOf entry order follows the shared registry (which is ordered for
        // the CLI help), so it may differ from the pre-registry hand-built
        // order; descriptions and structure are unchanged and the set is
        // pinned by `parameters_schema_has_all_actions`.
        let entries: Vec<Value> = crate::chrome::actions::ACTIONS
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
                    "description": "Logical name for this chrome session. \
                     Missing or empty uses the run's default tab, released when \
                     your run ends. Sessions live in a private per-run namespace, \
                     so a name here never addresses another run's session — and a \
                     run re-driven under its own identity re-attaches the names it \
                     was working with. Use an explicit name (e.g. \"docs\", \
                     \"github\") only to keep multiple pages open simultaneously. \
                     Same tab = serialized operations on that page."
                }
            },
            "required": ["action", "tab"]
        })
    }

    async fn execute(&self, _ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let (tab, action, normalized_notes) = Self::normalize_call(&args)?;

        debug!(tab, action = ?action, "chrome action");

        Self::ensure_available().await?;

        Self::validate_find(&action)?;

        // Get or create a per-tab lock — only serializes operations on the
        // same tab. Different tabs run fully concurrently.
        let _guard = self.acquire_tab_lock(&tab).await;

        if let ChromeAction::GetInnerText { selector } = &action {
            let output = self.get_inner_text(selector, &tab).await?;
            let body = if output.is_empty() {
                format!("[Tab: {tab}] (no output)")
            } else {
                format!("[Tab: {tab}] {output}")
            };
            return Ok(super::with_normalization_notes(body, &normalized_notes));
        }

        if let ChromeAction::Screenshot { .. } = &action {
            let output = self.capture_screenshot(&tab).await?;
            return Ok(super::with_normalization_notes(output, &normalized_notes));
        }

        if let ChromeAction::Extract { schema, limit } = &action {
            let (schema, schema_note) = Self::normalize_extract_schema(schema)?;
            let output = self.run_extract(&schema, *limit, &tab).await?;
            let mut notes = normalized_notes;
            notes.extend(schema_note);
            return Ok(super::with_normalization_notes(output, &notes));
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
        let action: ChromeAction = serde_json::from_value(action_value).ok()?;
        if !matches!(action, ChromeAction::Screenshot { .. }) {
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
            crate::tools::ImagePayloadSource::Chrome,
        ))
    }
}

impl ChromeTool {
    /// Resolve the tab for a call. An explicit non-empty tab passes through as the
    /// logical name; missing/empty falls back to the per-run default logical session,
    /// which [`ChromeTool::resolve_session`] maps into this run's namespace (a function
    /// of the agent id — see [`run_session_namespace`]). Defaulting is echoed in tool
    /// output (logical names only — physical session names never surface to the model).
    fn normalize_tab(args: &Value) -> (String, Option<String>) {
        if let Some(tab) = super::get_opt_str(args, "tab").filter(|s| !s.is_empty()) {
            (tab.to_string(), None)
        } else {
            let note = format!("missing/empty tab defaulted to \"{DEFAULT_TAB}\"");
            (DEFAULT_TAB.to_string(), Some(note))
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
                "Chrome screenshot reported success but no PNG was written at {}",
                path.display()
            );
        }
        // A corrupt/truncated PNG would otherwise emit a phantom `[IMAGE:…]`
        // marker that the payload path fails open on — gate the emission here.
        let (width, height) = validate_png(&path)?;
        *self.last_screenshot.lock().unwrap_poison() = Some(path_str.clone());
        Ok(format!(
            "[Tab: {tab}] Captured a chrome screenshot: {path_str} ({width}x{height}). \
             [IMAGE:{path_str}]"
        ))
    }

    /// Build a screenshot output path under the pinned temp root (or OS temp
    /// in tests) — one of the allowed temp roots. The filename is derived from
    /// a sanitized tab name plus a random nonce so a model-supplied `tab`
    /// (which is user-controlled) can never traverse out of the temp root.
    fn screenshot_output_path(tab: &str) -> anyhow::Result<PathBuf> {
        let dir = std::env::temp_dir().join("chrome-screenshots");
        std::fs::create_dir_all(&dir).with_context(|| {
            format!(
                "Failed to create chrome screenshot directory {}",
                dir.display()
            )
        })?;
        let slug = sanitize_filename_component(tab);
        let nonce = rand::random::<u64>();
        Ok(dir.join(format!("{slug}_{nonce:016x}.png")))
    }

    /// Normalize the raw tool arguments into the parsed action plus the tab
    /// and any normalization notes. LLM-facing corrective texts (action shape,
    /// find-hint) live with the tool, not the shared chrome core.
    fn normalize_call(args: &Value) -> anyhow::Result<(String, ChromeAction, Vec<String>)> {
        let mut normalized_notes: Vec<String> = Vec::new();
        let (tab, tab_note) = Self::normalize_tab(args);
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

        let action: ChromeAction = serde_json::from_value(action_value.clone()).map_err(|e| {
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
                "Invalid chrome action arguments{hint}. Expected action to be {EXPECTED_ACTION_SHAPE}, \
                 plus a \"tab\" string. Serde error: {e}"
            )
        })?;

        Ok((tab, action, normalized_notes))
    }

    /// Validate a `Find` action's locator type/action payload early for better
    /// diagnostics — before any CLI dispatch.
    fn validate_find(action: &ChromeAction) -> anyhow::Result<()> {
        if let ChromeAction::Find {
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

    /// Orchestrate a single chrome action: build the args, dispatch, and (for
    /// `Open`) the blank-navigation guard, the best-effort network-idle wait,
    /// and the compact auto-snapshot. Returns the RAW response and the snapshot
    /// text — no LLM-facing framing (that lives in [`Self::format_action_output`]).
    async fn run_action(
        &self,
        action: &ChromeAction,
        tab: &str,
    ) -> anyhow::Result<(ChromeResponse, String)> {
        let cli_args = Self::build_args(action)?;
        let str_args: Vec<&str> = cli_args.iter().map(String::as_str).collect();
        let response = self.run_command(&str_args, tab).await?;

        // A real navigation that ends on the scratch `about:blank` or Chrome's
        // error page never loaded — fail loudly (closing the blank-page tab
        // best-effort) instead of reporting success with no content.
        if let ChromeAction::Open { url } = action {
            self.bail_on_failed_navigation(tab, url, &response).await?;
        }

        // After open, wait for network idle, then auto-snapshot
        // so the LLM sees page content immediately.
        let snapshot = if matches!(action, ChromeAction::Open { .. }) {
            let wait_args = ["wait", "--load", "networkidle"];
            let _ = self.run_command(&wait_args, tab).await;

            // Compact snapshot to return page content; when it comes back
            // empty (canvas/PDF/SPA shells, capture failure) fall back to
            // visible text so open still yields something readable.
            let snap = self
                .run_command(&["snapshot", "-c"], tab)
                .await
                .ok()
                .and_then(|r| r.data.as_ref().and_then(extract_snapshot_text))
                .unwrap_or_default();
            if snap.trim().is_empty() {
                self.get_inner_text("body", tab).await.unwrap_or_default()
            } else {
                snap
            }
        } else {
            String::new()
        };

        Ok((response, snapshot))
    }

    /// Shape the raw [`ChromeResponse`] into the LLM-facing text: the
    /// per-action data extraction, the `[Tab: {tab}]` framing, and the
    /// normalization notes.
    fn format_action_output(
        action: &ChromeAction,
        tab: &str,
        response: ChromeResponse,
        snapshot: &str,
        notes: &[String],
    ) -> String {
        use std::fmt::Write as _;
        // chrome-use's degraded-success `warning` only applies to the
        // text-input actions (`type` read-back mismatch, `press` with no key
        // listeners / provably nowhere) — don't scan the shared path for them.
        let warning = if matches!(
            action,
            ChromeAction::Fill { .. } | ChromeAction::Type { .. } | ChromeAction::Press { .. }
        ) {
            crate::chrome::contract::chrome_use_warning(&response)
        } else {
            None
        };
        let output = match response.data {
            Some(data) => match action {
                ChromeAction::Snapshot { .. } | ChromeAction::GetText { .. } => {
                    extract_snapshot_text(&data)
                        .or_else(|| serde_json::to_string_pretty(&data).ok())
                        .unwrap_or_default()
                }
                ChromeAction::Open { .. } => {
                    let mut s = format!(
                        "Opened {}",
                        data.get("url").and_then(|v| v.as_str()).unwrap_or("?")
                    );
                    if snapshot.trim().is_empty() {
                        s.push_str("\n\n(no page content captured)");
                    } else {
                        let _ = write!(s, "\n\n--- Page content ---\n{snapshot}");
                    }
                    s
                }
                ChromeAction::GetUrl { .. } => data
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                ChromeAction::Eval { .. } => data.get("result").map_or_else(
                    || serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string()),
                    |v| v.as_str().map_or_else(|| v.to_string(), str::to_string),
                ),
                ChromeAction::Wait { .. } => "wait complete".to_string(),
                ChromeAction::Expect { .. } => match expect_outcome(&data) {
                    Some(outcome) if outcome.pass => "expect: PASS (condition holds)".to_string(),
                    Some(outcome) => {
                        let actual = outcome.actual.map_or("n/a".to_string(), |v| v.to_string());
                        let mut s = format!(
                            "expect: FAIL — condition did not hold (timed out: {}); actual: {actual}",
                            outcome.timed_out
                        );
                        // Same remediation the CLI expect envelope carries.
                        if outcome.timed_out {
                            s.push_str(EXPECT_TIMEOUT_NOTE);
                        }
                        s
                    }
                    None => {
                        serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string())
                    }
                },
                _ => serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string()),
            },
            None => String::new(),
        };

        let mut output = if output.is_empty() {
            format!("[Tab: {tab}] (no output)")
        } else {
            format!("[Tab: {tab}] {output}")
        };

        if let Some(warning) = warning {
            // The warning is normally a plain string — render it unquoted,
            // not as its JSON serialization.
            let text = warning
                .as_str()
                .map_or_else(|| warning.to_string(), str::to_string);
            let _ = write!(
                output,
                "\n\n⚠ chrome-use warning: {text} — treat the action as NOT applied and choose another approach"
            );
        }

        super::with_normalization_notes(output, notes)
    }

    /// Accept the model-facing extract schema: a JSON object, or a string
    /// holding JSON text (models frequently double-encode). The object must
    /// follow chrome-use's grammar — an optional `"rows"` selector string plus
    /// a required `"fields"` object (field name → CSS selector or
    /// `{sel, get, all}`). A flat schema with per-field keys at the top level
    /// is normalized into `"fields"` and the correction is reported as a note.
    /// Array-shaped objects (`{items: [...]}`) are rejected — wrapping them
    /// into `fields` would hand chrome-use a malformed schema.
    fn normalize_extract_schema(schema: &Value) -> anyhow::Result<(Value, Option<String>)> {
        let parsed = match schema {
            Value::Object(_) => schema.clone(),
            Value::String(s) => serde_json::from_str::<Value>(s).map_err(|_| {
                anyhow::anyhow!(
                    "extract schema string is not valid JSON — pass the schema as a JSON object"
                )
            })?,
            _ => anyhow::bail!(
                "extract schema must be a JSON object with a \"fields\" object (field name → CSS selector)"
            ),
        };
        let Some(map) = parsed.as_object() else {
            anyhow::bail!(
                "extract schema must be a JSON object with a \"fields\" object (field name → CSS selector)"
            )
        };
        let field_shape = "field name → CSS selector string or {sel, get, all} object";
        if let Some(rows) = map.get("rows") {
            anyhow::ensure!(
                rows.is_string(),
                "extract schema \"rows\" must be a CSS selector string"
            );
        }
        let (out, note) = if let Some(fields) = map.get("fields") {
            anyhow::ensure!(
                fields.is_object(),
                "extract schema \"fields\" must be an object ({field_shape})"
            );
            (parsed, None)
        } else {
            let mut fields = serde_json::Map::new();
            for (k, v) in map.iter().filter(|(k, _)| k.as_str() != "rows") {
                anyhow::ensure!(
                    v.is_string() || v.is_object(),
                    "extract schema field '{k}' must be a {field_shape} — \
                     chrome-use requires a \"fields\" object"
                );
                fields.insert(k.clone(), v.clone());
            }
            let mut out = serde_json::Map::new();
            if let Some(r) = map.get("rows") {
                out.insert("rows".into(), r.clone());
            }
            out.insert("fields".into(), Value::Object(fields));
            (
                Value::Object(out),
                Some(
                    "flat extract schema normalized: field keys moved under \"fields\" \
                     (chrome-use requires a fields object)"
                        .to_string(),
                ),
            )
        };
        // The getter vocabulary is independent of the shape (object vs flat);
        // validate the final normalized schema exactly once so a bad getter is
        // rejected loudly instead of chrome-use's silent textContent fallback.
        validate_extract_getters(&out).map_err(anyhow::Error::msg)?;
        Ok((out, note))
    }

    /// Schema-driven extraction with the honest-empty gate: when the schema's
    /// rows selector matches 0 elements, report the empty region WITHOUT
    /// invoking chrome-use's phantom-row extract (its dominant-container
    /// auto-detect returns garbage rows on a 0-match, and an invalid CSS
    /// rows selector triggers the same fallback). A failing count eval is
    /// an explicit error — the fallback is deliberately not used. `limit`
    /// is mahbot-side post-filtering (chrome-use has no extract limit key);
    /// `total` stays honest. `schema` is already normalized
    /// ([`Self::normalize_extract_schema`]).
    async fn run_extract(
        &self,
        schema: &Value,
        limit: Option<usize>,
        tab: &str,
    ) -> anyhow::Result<String> {
        let rows_sel = schema.get("rows").and_then(Value::as_str);
        if let Some(sel) = rows_sel {
            let js = count_eval_js(sel);
            let resp = self.run_command(&["eval", &js], tab).await?;
            let count = eval_count(&resp).ok_or_else(|| {
                anyhow::anyhow!(
                    "count eval for the rows selector '{sel}' returned a non-numeric result"
                )
            });
            match extract_gate(count) {
                ExtractGate::Empty => {
                    return serde_json::to_string_pretty(&extract_output(
                        &Value::Array(vec![]),
                        limit,
                    ))
                    .context("serialize extract output");
                }
                ExtractGate::Proceed => {}
                ExtractGate::Fail(e) => return Err(e),
            }
        }
        let schema_str = serde_json::to_string(schema)?;
        let resp = self
            .run_command(&["extract", "--schema", &schema_str], tab)
            .await?;
        let Some(data) = resp.data.as_ref() else {
            anyhow::bail!("extract returned no data");
        };
        serde_json::to_string_pretty(&extract_output(data, limit))
            .context("serialize extract output")
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
/// `-`, `_`), so a model-supplied value (e.g. a chrome `tab`) can never
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

/// Outcome of parsing one chrome-use CLI output in the tool (see
/// [`parse_run_output`]).
#[derive(Debug)]
enum ParsedRun {
    Ok(ChromeResponse),
    /// Failure envelope (or unparseable stdout on a non-zero exit): the raw
    /// error text (empty when the envelope carried none), envelope code, and
    /// retryable flag.
    Failed {
        error: String,
        code: Option<String>,
        retryable: Option<bool>,
    },
    /// Zero exit but unparseable stdout.
    Unparseable,
}

/// Parse one chrome-use CLI output. `expect` is the one chrome-use action
/// whose `--json` envelope can succeed on a non-zero exit — a failed
/// assertion arrives as `success:true` with exit 1, the verdict riding in
/// `data.pass` — so envelope-success is trusted over the exit code there and
/// nowhere else. Pure so tests pin the mapping.
fn parse_run_output(
    action: Option<&str>,
    status_success: bool,
    stdout: &str,
    stderr: &str,
) -> ParsedRun {
    let parsed: Option<ChromeResponse> = serde_json::from_str(stdout).ok();
    let Some(resp) = parsed else {
        return if status_success {
            ParsedRun::Unparseable
        } else {
            ParsedRun::Failed {
                error: stderr.trim().to_string(),
                code: None,
                retryable: None,
            }
        };
    };
    if resp.is_success() && (status_success || action == Some("expect")) {
        return ParsedRun::Ok(resp);
    }
    // A failure envelope (or, for a non-expect action, the unusable
    // success-on-non-zero-exit combination) — surface its error details.
    ParsedRun::Failed {
        error: resp.error.unwrap_or_default(),
        code: resp.code,
        retryable: resp.retryable,
    }
}

/// Enhance chrome-use error messages with actionable hints for known
/// failure patterns.
fn enhance_chrome_error(msg: String) -> String {
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

/// Known chrome action variant names (must match `ChromeAction` serde names).
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
    "fill",
    "type",
    "eval",
    "find",
    "screenshot",
    "wait",
    "expect",
    "extract",
];

/// Expected action shape, echoed verbatim in corrective errors so the model
/// can self-correct in one round-trip.
const EXPECTED_ACTION_SHAPE: &str = "one of: {\"open\":{\"url\":\"https://...\"}}, \
    {\"snapshot\":{\"interactive_only\":bool,\"compact\":bool,\"depth\":int}}, \
    {\"click\":{\"selector\":\"...\"}}, {\"get_text\":{\"selector\":\"...\"}}, \
    {\"get_innertext\":{\"selector\":\"...\"}}, {\"get_url\":{}}, \
    {\"press\":{\"key\":\"...\",\"selector\":\"...\"}}, \
    {\"fill\":{\"selector\":\"...\",\"text\":\"...\"}}, \
    {\"type\":{\"selector\":\"...\",\"text\":\"...\",\"key_events\":bool}}, \
    {\"eval\":{\"js\":\"...\"}}, \
    {\"find\":{\"by\":\"text|role|label|placeholder|alt|title|testid|first|last|nth\",\
    \"value\":\"...\",\"action\":\"click|fill|hover|check|text\"}}, \
    {\"screenshot\":{}}, {\"wait\":{\"selector\":\"...\"}}, \
    {\"expect\":{\"condition\":\"visible|hidden|present|count|text|value|attr|url\",\
    \"selector\":\"...\",\"count\":int,\"op\":\"==\",\"predicate\":\"equals|contains|matches\",\"name\":\"...\",\"expected\":\"...\"}}, \
    {\"extract\":{\"schema\":{...},\"limit\":int}}";

/// Corrective error for an unrecoverable action shape, listing the exact
/// expected form instead of raw serde text.
fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid chrome action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}, \
         plus a \"tab\" string. Received: {received}"
    )
}

/// Normalize a model-supplied chrome `action` value into the canonical
/// `{"variant": {...}}` tagged form that `ChromeAction` deserializes from.
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
    use crate::tools::chrome_daemon::chrome_bin;

    // ── build_args: simple actions ───────────────────────────────────────

    #[test]
    fn build_args_for_simple_actions() {
        struct Case {
            name: &'static str,
            action: ChromeAction,
            expected: &'static [&'static str],
        }

        let cases = [
            Case {
                name: "open",
                action: ChromeAction::Open {
                    url: "https://example.com".into(),
                },
                expected: &["open", "https://example.com"],
            },
            Case {
                name: "snapshot",
                action: ChromeAction::Snapshot {
                    interactive_only: true,
                    compact: true,
                    depth: Some(5),
                },
                expected: &["snapshot", "-i", "-c", "-d", "5"],
            },
            Case {
                name: "click",
                action: ChromeAction::Click {
                    selector: "@e1".into(),
                },
                expected: &["click", "@e1"],
            },
            Case {
                name: "get_text",
                action: ChromeAction::GetText {
                    selector: "@e3".into(),
                },
                expected: &["get", "text", "@e3"],
            },
            Case {
                name: "get_url",
                action: ChromeAction::GetUrl {},
                expected: &["get", "url"],
            },
            Case {
                name: "fill",
                action: ChromeAction::Fill {
                    selector: "#e".into(),
                    text: "hi".into(),
                },
                expected: &["fill", "#e", "hi"],
            },
            Case {
                name: "fill leading-dash text",
                action: ChromeAction::Fill {
                    selector: "#e".into(),
                    text: "--foo".into(),
                },
                expected: &["fill", "#e", "--", "--foo"],
            },
            Case {
                name: "type",
                action: ChromeAction::Type {
                    selector: "#e".into(),
                    text: "hi".into(),
                    key_events: true,
                },
                expected: &["type", "#e", "--key-events", "hi"],
            },
            Case {
                name: "type leading-dash text",
                action: ChromeAction::Type {
                    selector: "#e".into(),
                    text: "--foo".into(),
                    key_events: true,
                },
                // The shield goes last: everything after `--` is a verbatim
                // value, so --key-events must precede it.
                expected: &["type", "#e", "--key-events", "--", "--foo"],
            },
            Case {
                name: "press no selector",
                action: ChromeAction::Press {
                    key: "Enter".into(),
                    selector: None,
                },
                expected: &["press", "Enter"],
            },
            Case {
                name: "press with selector",
                action: ChromeAction::Press {
                    key: "Enter".into(),
                    selector: Some("#t".into()),
                },
                expected: &["press", "Enter", "--selector", "#t"],
            },
        ];

        for case in &cases {
            let args = ChromeTool::build_args(&case.action).unwrap_or_else(|e| {
                panic!("{}: build_args failed: {}", case.name, e);
            });
            assert_eq!(args, case.expected, "{}", case.name);
        }
    }

    // ── build_args: GetInnerText error ──────────────────────────────────

    #[test]
    fn build_args_rejects_get_innertext() {
        let action = ChromeAction::GetInnerText {
            selector: "body".into(),
        };
        assert!(
            ChromeTool::build_args(&action).is_err(),
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
            let action = ChromeAction::Find {
                by: case.by.into(),
                value: case.value.into(),
                action: case.action.into(),
                text: case.text.map(String::from),
                name: case.find_name.map(String::from),
                exact: case.exact,
                index: case.index,
            };
            let args = ChromeTool::build_args(&action).unwrap_or_else(|e| {
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
            let action = ChromeAction::Find {
                by: "text".into(),
                value: "anything".into(),
                action: bad_action.into(),
                text: None,
                name: None,
                exact: None,
                index: None,
            };
            let err = ChromeTool::validate_find(&action)
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
        let find = |by: &str, name: Option<&str>, exact: Option<bool>| ChromeAction::Find {
            by: by.into(),
            value: ".card".into(),
            action: "click".into(),
            text: None,
            name: name.map(String::from),
            exact,
            index: if by == "nth" { Some(0) } else { None },
        };
        for by in ["nth", "first", "last", "testid"] {
            let err = ChromeTool::validate_find(&find(by, Some("Submit"), None))
                .expect_err("name with CSS-selector locator must be rejected")
                .to_string();
            assert!(
                err.contains("`name`/`exact`") && err.contains(by),
                "by {by}: {err}"
            );
            let err = ChromeTool::validate_find(&find(by, None, Some(true)))
                .expect_err("exact with CSS-selector locator must be rejected")
                .to_string();
            assert!(
                err.contains("`name`/`exact`") && err.contains(by),
                "by {by}: {err}"
            );
        }
        // role-style locators still honor name/exact.
        ChromeTool::validate_find(&find("role", Some("Submit"), Some(false)))
            .expect("role + name/exact must still validate");
    }

    // ── call_timeout policy ──────────────────────────────────────────────

    #[test]
    fn call_timeout_policy() {
        assert_eq!(
            ChromeTool::call_timeout(&["open", "https://example.com"]),
            Duration::from_secs(22)
        );
        assert_eq!(
            ChromeTool::call_timeout(&["wait", "--load", "networkidle"]),
            Duration::from_secs(12)
        );
        assert_eq!(
            ChromeTool::call_timeout(&["expect", "visible", "#x"]),
            Duration::from_secs(22)
        );
        for args in [
            &["click", "@e1"][..],
            &["eval", "1+1"][..],
            &["snapshot", "-c"][..],
        ] {
            assert_eq!(
                ChromeTool::call_timeout(args),
                Duration::from_secs(8),
                "args: {args:?}"
            );
        }
    }

    #[test]
    fn tool_name_and_description_are_set() {
        let tool = ChromeTool::default();
        assert_eq!(tool.name(), "chrome");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn parameters_schema_is_valid_json() {
        let tool = ChromeTool::default();
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
        let tool = ChromeTool::default();
        let schema = tool.parameters_schema();
        let action_schemas = schema["properties"]["action"]["oneOf"]
            .as_array()
            .expect("oneOf should be an array");

        // There are exactly 15 chrome actions.
        assert_eq!(
            action_schemas.len(),
            15,
            "expected 15 actions, got {}",
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
            "fill",
            "type",
            "eval",
            "find",
            "screenshot",
            "wait",
            "expect",
            "extract",
        ] {
            assert!(
                action_names.contains(expected),
                "schema missing action: {expected}"
            );
        }

        // Structural invariants: snapshot, get_url, screenshot and wait must lack
        // inner "required"; all other actions must have it.
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
            if name == "snapshot" || name == "get_url" || name == "screenshot" || name == "wait" {
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
    fn chrome_bin_name_is_correct() {
        let name = chrome_bin();
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
        let tool = ChromeTool::default();

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
    fn missing_or_empty_tab_defaults_to_logical_default() {
        // Missing tab → the stable logical default, echoed in tool output.
        // (Per-run physical uniqueness of that default is covered by
        // session_resolution_namespaces_per_agent_run.)
        let (tab, note) =
            ChromeTool::normalize_tab(&json!({"action":{"open":{"url":"https://example.com"}}}));
        assert_eq!(tab, DEFAULT_TAB);
        assert!(note.is_some(), "defaulting should be echoed in tool output");
        // Empty tab → the same default.
        let (tab2, note) =
            ChromeTool::normalize_tab(&json!({"tab":"","action":{"open":{"url":"x"}}}));
        assert_eq!(tab, tab2);
        assert!(note.is_some());
        // Explicit tab passes through as the logical name.
        let (tab, note) =
            ChromeTool::normalize_tab(&json!({"tab":"docs","action":{"open":{"url":"x"}}}));
        assert_eq!(tab, "docs");
        assert!(note.is_none());
    }

    #[test]
    fn session_resolution_namespaces_per_agent_run() {
        // Two agent runs using the SAME logical tabs resolve to DIFFERENT
        // physical sessions — no cross-run drift or contamination.
        let a = ChromeTool::new(ChromeRunSessions::for_run("run-a"));
        let b = ChromeTool::new(ChromeRunSessions::for_run("run-b"));
        for tab in [DEFAULT_TAB, "docs"] {
            let (sa, sb) = (a.resolve_session(tab), b.resolve_session(tab));
            assert_ne!(sa, sb, "each run needs its own session for tab {tab}");
            assert!(sa.starts_with(AGENT_TAB_PREFIX) && sb.starts_with(AGENT_TAB_PREFIX));
            assert!(sa.ends_with(tab) && sb.ends_with(tab));
        }
        // The namespace is a pure function of the run's agent id: a run
        // re-derived after a daemon restart (or a self-update) resolves the
        // same physical names, so it re-attaches to the tabs it was working
        // with and its release addresses them.
        let resumed = ChromeTool::new(ChromeRunSessions::for_run("run-a"));
        assert_eq!(resumed.resolve_session("docs"), a.resolve_session("docs"));
        // Idempotent: echoing back an own physical name never double-prefixes.
        let physical = a.resolve_session("docs");
        assert_eq!(a.resolve_session(&physical), physical);
        // Another run's physical name still lands in THIS run's namespace —
        // it can never resolve to someone else's session.
        let other_physical = b.resolve_session("docs");
        assert_ne!(a.resolve_session(&other_physical), other_physical);
        // The shared non-agent instance passes names through unchanged —
        // link enrichment keeps its own daemon-sweepable namespace.
        assert_eq!(
            ChromeTool::default().resolve_session("link-enricher-3"),
            "link-enricher-3"
        );
        // Model-supplied tabs are sanitized into chrome-use's charset.
        let s = a.resolve_session("my tab/../x");
        assert!(
            s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "physical session must be charset-safe, got {s}"
        );
        // Distinct logical names never collapse into one session even when
        // the sanitizer maps them to the same safe string.
        assert_ne!(a.resolve_session("a b"), a.resolve_session("a/b"));
    }

    #[test]
    fn agent_sessions_are_never_orphan_swept() {
        // The namespace keeps the protected `agent-tab-` root, so the
        // daemon's orphan-protection sweep can never close a live run's
        // session; the run's own release (verified `session stop`) reclaims
        // them, not this sweep.
        let tool = ChromeTool::new(ChromeRunSessions::for_run("run-namespace"));
        for tab in [DEFAULT_TAB, "docs"] {
            assert!(!crate::tools::chrome_daemon::is_mahbot_session_name(
                &tool.resolve_session(tab)
            ));
        }
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
        let _guard = crate::tools::chrome_daemon::with_health_test_lock().await;
        // Pristine start (a sibling health test may have left a Down fixture).
        crate::tools::chrome_daemon::reset_health();
        // An orphaned-tab error (even envelope-wrapped with the auto-connect
        // and daemon-wrapper text) fails fast with hand-close guidance but
        // does NOT mark the daemon unhealthy — the relay and daemon are up, so
        // recovery must not wake for it.
        let err = ChromeTool::fail_fast_if_daemon_down(
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
        assert!(crate::tools::chrome_daemon::is_advertised());
        // Daemon-unavailable signature → actionable guidance, daemon marked
        // unhealthy (wakes the auto-recovery watchdog).
        let err = ChromeTool::fail_fast_if_daemon_down(
            "Failed to read: Resource temporarily unavailable (os error 35) (after 5 retries - daemon may be busy or unresponsive)",
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("chrome daemon is down"),
            "expected guidance message, got: {err}"
        );
        assert!(!crate::tools::chrome_daemon::is_advertised());
        // The unambiguous daemon-side envelope code fails fast too (no message
        // matching).
        let err = ChromeTool::fail_fast_if_daemon_down(
            "chrome-use error: browser not launched",
            Some("browser_not_launched"),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("chrome daemon is down"),
            "expected guidance message, got: {err}"
        );
        // Site-level failures — even ones carrying the coarse `connection_failed`
        // code (the CLI assigns it to any "connection" text) — pass through as
        // truthful navigation failures and leave the daemon healthy.
        assert!(
            ChromeTool::fail_fast_if_daemon_down(
                "chrome-use error: Navigation failed: net::ERR_CONNECTION_REFUSED",
                Some("connection_failed"),
            )
            .is_ok()
        );
        assert!(
            ChromeTool::fail_fast_if_daemon_down("chrome-use error: Element not found", None)
                .is_ok()
        );
        assert!(
            ChromeTool::fail_fast_if_daemon_down("chrome-use error: timed out", Some("timeout"))
                .is_ok()
        );
        // Restore the global health singleton so later agent-constructing
        // tests don't inherit a hidden chrome tool.
        crate::tools::chrome_daemon::reset_health();
    }

    // -----------------------------------------------------------------------
    // KNOWN_ACTIONS lockstep — adding a ChromeAction variant must be mirrored
    // in the normalization allowlist.
    // -----------------------------------------------------------------------

    #[test]
    fn known_actions_lockstep_with_chrome_action_variants() {
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
                "press" => json!({"key": "Enter", "selector": "#t"}),
                "fill" => json!({"selector": "#e", "text": "hi"}),
                "type" => json!({"selector": "#e", "text": "hi", "key_events": true}),
                "eval" => json!({"js": "1 + 1"}),
                "find" => json!({"by": "text", "value": "x", "action": "click"}),
                "wait" => json!({"selector": "#x"}),
                "expect" => json!({"condition": "visible", "selector": "#x"}),
                "extract" => json!({"schema": {"rows": ".r"}}),
                other => panic!("KNOWN_ACTIONS entry {other} has no lockstep payload"),
            }
        };
        // Every allowlist entry must be a real variant name — stale entries
        // (e.g. invented actions) fail deserialization here.
        for name in KNOWN_ACTIONS {
            let tagged = json!({*name: payload(name)});
            assert!(
                serde_json::from_value::<ChromeAction>(tagged.clone()).is_ok(),
                "KNOWN_ACTIONS entry does not deserialize: {tagged}"
            );
        }
        // Every variant's canonical name must be in the allowlist — a new
        // ChromeAction variant fails here.
        for name in [
            "open",
            "snapshot",
            "click",
            "get_text",
            "get_inner_text",
            "get_url",
            "press",
            "fill",
            "type",
            "eval",
            "find",
            "screenshot",
            "wait",
            "expect",
            "extract",
        ] {
            assert!(
                KNOWN_ACTIONS.contains(&name),
                "KNOWN_ACTIONS is missing variant {name}"
            );
        }
    }

    #[test]
    fn build_args_rejects_screenshot() {
        let action = ChromeAction::Screenshot {};
        assert!(
            ChromeTool::build_args(&action).is_err(),
            "Screenshot must be handled in execute(), not build_args"
        );
    }

    #[expect(clippy::too_many_lines)] // the per-argv wait/expect shapes are one cohesive table
    #[test]
    fn build_args_for_wait_and_expect() {
        fn argv(args: &[String]) -> Vec<&str> {
            args.iter().map(String::as_str).collect()
        }

        // Wait: exactly one target; the forwarded --timeout is the declared
        // chrome-side deadline (the mahbot kill rides DEADLINE_SLACK above it).
        let wait_selector = ChromeAction::Wait {
            selector: Some("#r".into()),
            url: None,
            text: None,
        };
        let got = ChromeTool::build_args(&wait_selector).unwrap();
        assert_eq!(argv(&got), vec!["wait", "#r", "--timeout", "10000"]);

        let wait_url = ChromeAction::Wait {
            selector: None,
            url: Some("dashboard".into()),
            text: None,
        };
        let got = ChromeTool::build_args(&wait_url).unwrap();
        assert_eq!(
            argv(&got),
            vec!["wait", "--url", "dashboard", "--timeout", "10000"]
        );

        let wait_text = ChromeAction::Wait {
            selector: None,
            url: None,
            text: Some("Loaded".into()),
        };
        let got = ChromeTool::build_args(&wait_text).unwrap();
        assert_eq!(
            argv(&got),
            vec!["wait", "--text", "Loaded", "--timeout", "10000"]
        );

        // Wait requires exactly one target.
        let both = ChromeAction::Wait {
            selector: Some("#r".into()),
            url: Some("dashboard".into()),
            text: None,
        };
        assert!(
            ChromeTool::build_args(&both).is_err(),
            "two targets must err"
        );
        let none = ChromeAction::Wait {
            selector: None,
            url: None,
            text: None,
        };
        assert!(ChromeTool::build_args(&none).is_err(), "no target must err");
        // A numeric selector would be chrome-use's silent-sleep form — rejected.
        let numeric = ChromeAction::Wait {
            selector: Some("5000".into()),
            url: None,
            text: None,
        };
        let err = ChromeTool::build_args(&numeric).unwrap_err().to_string();
        assert!(err.contains("silent sleep"), "err: {err}");

        // Expect visible.
        let expect_visible = ChromeAction::Expect {
            condition: "visible".into(),
            selector: Some("#main".into()),
            op: None,
            count: None,
            predicate: None,
            name: None,
            expected: None,
        };
        let got = ChromeTool::build_args(&expect_visible).unwrap();
        assert_eq!(
            argv(&got),
            vec!["expect", "#main", "visible", "--timeout", "20000"]
        );

        // Expect count defaults op to ==.
        let expect_count_default = ChromeAction::Expect {
            condition: "count".into(),
            selector: Some(".card".into()),
            op: None,
            count: Some(5),
            predicate: None,
            name: None,
            expected: None,
        };
        let got = ChromeTool::build_args(&expect_count_default).unwrap();
        assert_eq!(
            argv(&got),
            vec!["expect", "count", ".card", "==", "5", "--timeout", "20000"]
        );

        // Expect count forwards an explicit op.
        let expect_count_op = ChromeAction::Expect {
            condition: "count".into(),
            selector: Some(".card".into()),
            op: Some(">=".into()),
            count: Some(5),
            predicate: None,
            name: None,
            expected: None,
        };
        let got = ChromeTool::build_args(&expect_count_op).unwrap();
        assert_eq!(
            argv(&got),
            vec!["expect", "count", ".card", ">=", "5", "--timeout", "20000"]
        );

        // Expect url with a predicate.
        let expect_url = ChromeAction::Expect {
            condition: "url".into(),
            selector: None,
            op: None,
            count: None,
            predicate: Some("contains".into()),
            name: None,
            expected: Some("dashboard".into()),
        };
        let got = ChromeTool::build_args(&expect_url).unwrap();
        assert_eq!(
            argv(&got),
            vec![
                "expect",
                "url",
                "contains",
                "dashboard",
                "--timeout",
                "20000"
            ]
        );

        // Expect corrective errors.
        let bad_condition = ChromeAction::Expect {
            condition: "banana".into(),
            selector: None,
            op: None,
            count: None,
            predicate: None,
            name: None,
            expected: None,
        };
        assert!(ChromeTool::build_args(&bad_condition).is_err());

        let count_no_count = ChromeAction::Expect {
            condition: "count".into(),
            selector: Some(".card".into()),
            op: None,
            count: None,
            predicate: None,
            name: None,
            expected: None,
        };
        assert!(ChromeTool::build_args(&count_no_count).is_err());

        let visible_no_selector = ChromeAction::Expect {
            condition: "visible".into(),
            selector: None,
            op: None,
            count: None,
            predicate: None,
            name: None,
            expected: None,
        };
        assert!(ChromeTool::build_args(&visible_no_selector).is_err());

        let bad_op = ChromeAction::Expect {
            condition: "count".into(),
            selector: Some(".card".into()),
            op: Some("!%".into()),
            count: Some(5),
            predicate: None,
            name: None,
            expected: None,
        };
        let err = ChromeTool::build_args(&bad_op).unwrap_err().to_string();
        assert!(err.contains("invalid count op"), "err: {err}");

        let bad_predicate = ChromeAction::Expect {
            condition: "text".into(),
            selector: Some("h1".into()),
            op: None,
            count: None,
            predicate: Some("bogus".into()),
            name: None,
            expected: Some("x".into()),
        };
        let err = ChromeTool::build_args(&bad_predicate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid predicate"), "err: {err}");
    }

    /// Direct pin of the run-level classification, including the crux of the
    /// expect contract: a failed assertion arrives as `success:true` with
    /// exit 1 and the verdict in `data`.
    #[test]
    fn parse_run_output_pins_envelope_over_exit_code() {
        // expect pass=false: success:true + exit 1 → the verdict is returned.
        let body = json!({"success": true, "data": {"pass": false, "actual": 3, "timedOut": true}});
        let resp = match parse_run_output(Some("expect"), false, &body.to_string(), "") {
            ParsedRun::Ok(resp) => resp,
            other => panic!("expected Ok, got {other:?}"),
        };
        let outcome = expect_outcome(resp.data.as_ref().expect("data present")).expect("verdict");
        assert!(!outcome.pass && outcome.timed_out);
        assert_eq!(outcome.actual, Some(json!(3)));

        // …but only for expect: the same payload on another action is a failure.
        assert!(matches!(
            parse_run_output(Some("open"), false, &body.to_string(), ""),
            ParsedRun::Failed { .. }
        ));

        // A failure envelope on any exit → Failed with its details.
        let err_body = json!({"success": false, "error": "Browser not launched", "code": "browser_not_launched"});
        match parse_run_output(Some("expect"), false, &err_body.to_string(), "") {
            ParsedRun::Failed { error, code, .. } => {
                assert_eq!(error, "Browser not launched");
                assert_eq!(code.as_deref(), Some("browser_not_launched"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }

        // Unparseable stdout: zero exit → Unparseable; non-zero → stderr fallback.
        assert!(matches!(
            parse_run_output(None, true, "garbage", ""),
            ParsedRun::Unparseable
        ));
        match parse_run_output(None, false, "garbage", "relay is down\n") {
            ParsedRun::Failed {
                error,
                code,
                retryable,
            } => {
                assert_eq!(error, "relay is down");
                assert!(code.is_none() && retryable.is_none());
            }
            other => panic!("expected Failed, got {other:?}"),
        }

        // Happy path.
        assert!(matches!(
            parse_run_output(
                Some("open"),
                true,
                &json!({"success": true}).to_string(),
                ""
            ),
            ParsedRun::Ok(_)
        ));
    }

    #[test]
    fn normalize_extract_schema_shapes() {
        // A chrome-use-grammar object (rows + fields) passes through, no note.
        let obj = json!({"rows": ".card", "fields": {"title": ".title"}});
        let (out, note) = ChromeTool::normalize_extract_schema(&obj).unwrap();
        assert_eq!(out, obj);
        assert!(note.is_none());

        // A flat schema (per-field keys at the top level) is normalized into
        // the required "fields" object, with a correction note.
        let (out, note) =
            ChromeTool::normalize_extract_schema(&json!({"rows": ".card", "title": ".title"}))
                .unwrap();
        assert_eq!(out, json!({"rows": ".card", "fields": {"title": ".title"}}));
        assert!(note.is_some_and(|n| n.contains("fields")));

        // A string holding valid JSON text parses to the object.
        let stringified = json!("{\"rows\": \".card\", \"fields\": {}}");
        let (out, note) = ChromeTool::normalize_extract_schema(&stringified).unwrap();
        assert_eq!(out, json!({"rows": ".card", "fields": {}}));
        assert!(note.is_none());

        // A non-JSON string errs with the corrective message.
        let err = ChromeTool::normalize_extract_schema(&json!("not json"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not valid JSON"), "err: {err}");

        // A non-object shape errs.
        assert!(ChromeTool::normalize_extract_schema(&json!(42)).is_err());

        // An array-shaped object (no fields key, array values) errs instead of
        // being mangled into a fields object.
        let err = ChromeTool::normalize_extract_schema(&json!({"items": [1, 2]}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be a"), "err: {err}");

        // A "fields" key that is not an object errs.
        let err = ChromeTool::normalize_extract_schema(&json!({"fields": ".title"}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"fields\" must be an object"), "err: {err}");

        // Getters are validated post-normalization: a bare attribute name (no
        // "@") is rejected loudly instead of chrome-use's silent textContent
        // fallback.
        let err = ChromeTool::normalize_extract_schema(&json!({
            "fields": {"link": {"sel": "a", "get": "href"}}
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains('@'), "err: {err}");
        assert!(err.contains("getter"), "err: {err}");

        // An "@"-prefixed attribute getter is accepted.
        assert!(
            ChromeTool::normalize_extract_schema(&json!({
                "fields": {"u": {"sel": "a", "get": "@href"}}
            }))
            .is_ok()
        );
    }

    #[test]
    fn screenshot_action_normalizes_and_parses() {
        // Canonical tagged object.
        let action = serde_json::json!({"screenshot": {}});
        let parsed: ChromeAction = serde_json::from_value(action.clone()).unwrap();
        assert!(matches!(parsed, ChromeAction::Screenshot {}));

        // Plain action name with a tab sibling.
        let args = serde_json::json!({"action": "screenshot", "tab": "docs"});
        let (normalized, note) =
            normalize_action(args.get("action").cloned().unwrap(), &args).unwrap();
        assert!(
            note.is_some(),
            "screenshot should record a normalization note"
        );
        let parsed: ChromeAction = serde_json::from_value(normalized).unwrap();
        assert!(matches!(parsed, ChromeAction::Screenshot {}));
    }

    #[test]
    #[expect(clippy::case_sensitive_file_extension_comparisons)] // the tool itself always emits a lowercase .png name
    fn screenshot_output_path_is_safe_and_unique() {
        // A hostile tab name must not escape the temp dir.
        let p = ChromeTool::screenshot_output_path("../../etc/cron.d").unwrap();
        let file_name = p.file_name().unwrap().to_string_lossy().into_owned();
        assert!(!file_name.contains('/') && !file_name.contains('\\'));
        assert!(file_name.ends_with(".png"));
        // Two calls yield different files (random nonce).
        let p2 = ChromeTool::screenshot_output_path("default").unwrap();
        assert_ne!(p, p2);
    }

    // ── validate_png ─────────────────────────────────────────────────────

    #[test]
    fn validate_png_accepts_real_png() {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let dir = std::env::temp_dir().join("mahbot-chrome-test-valid");
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
        let dir = std::env::temp_dir().join("mahbot-chrome-test-corrupt");
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
        let response = ChromeResponse {
            data: Some(json!({"origin": "https://x", "result": "hello"})),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::format_action_output(
            &ChromeAction::Eval { js: "42".into() },
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
        let response = ChromeResponse {
            data: Some(json!({"note": "no result key"})),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::format_action_output(
            &ChromeAction::Eval { js: "42".into() },
            "default",
            response,
            "",
            &[],
        );
        assert!(out.contains('{'), "should fall back to pretty-print: {out}");
    }

    #[test]
    fn format_action_output_appends_warning_for_type_read_back_mismatch() {
        // chrome-use's `type` warning on a read-back mismatch — exit 0, so the
        // tool must flag it.
        let response = ChromeResponse {
            data: Some(json!({
                "typed": "hi", "readBack": "h",
                "warning": "the field does not contain what was typed",
            })),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::format_action_output(
            &ChromeAction::Type {
                selector: "#e".into(),
                text: "hi".into(),
                key_events: false,
            },
            "default",
            response,
            "",
            &[],
        );
        assert!(
            out.contains("chrome-use warning"),
            "output should surface the warning: {out}"
        );
        // The warning is rendered unquoted, not as its JSON serialization.
        assert!(
            out.contains("warning: the field does not contain what was typed —"),
            "warning text should be rendered unquoted: {out}"
        );
        assert!(
            out.contains("the field does not contain what was typed"),
            "output should carry the warning text: {out}"
        );
        assert!(
            out.contains("NOT applied"),
            "output should tell the agent to treat the action as not applied: {out}"
        );
    }

    /// `readBack` without a `warning` is chrome-use's normal success shape —
    /// no note may be appended.
    #[test]
    fn format_action_output_clean_response_has_no_warning() {
        let response = ChromeResponse {
            data: Some(json!({"typed": "hi", "readBack": "hi"})),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::format_action_output(
            &ChromeAction::Type {
                selector: "#e".into(),
                text: "hi".into(),
                key_events: false,
            },
            "default",
            response,
            "",
            &[],
        );
        assert!(
            !out.contains("chrome-use warning"),
            "clean output must not carry a warning: {out}"
        );
    }

    #[tokio::test]
    async fn image_payload_attaches_chrome_screenshot() {
        // Build a tiny PNG on disk.
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let dir = std::env::temp_dir().join("mahbot-chrome-test");
        std::fs::create_dir_all(&dir).unwrap();
        let png_path = dir.join("shot.png");
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&png_path, &buf).unwrap();

        let tool = ChromeTool::default();
        *tool.last_screenshot.lock().unwrap_poison() =
            Some(png_path.to_string_lossy().into_owned());

        let payload = tool
            .image_payload(
                &crate::Workspace::default(),
                &serde_json::json!({"action": "screenshot", "tab": "default"}),
            )
            .await
            .expect("screenshot must produce an image payload");
        assert_eq!(payload.source, crate::tools::ImagePayloadSource::Chrome);
        assert_eq!(payload.format, "PNG");
        assert!(payload.data_uri.starts_with("data:image/jpeg;base64,"));
        assert!(
            payload
                .attached_annotation()
                .starts_with("Chrome screenshot"),
            "annotation must describe it as a chrome screenshot: {}",
            payload.attached_annotation()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn image_payload_ignores_non_screenshot_actions() {
        let tool = ChromeTool::default();
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

        // Envelope-verdict gating lives in the caller (close_all_chrome_sessions_inner);
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
