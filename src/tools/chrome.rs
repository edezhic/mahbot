//! Chrome automation tool.

use crate::chrome::contract::{
    ChromeResponse, ERROR_PAGE_PROBE_JS, EXPECT_TIMEOUT_NOTE, chrome_leftover_pipe_note,
    chrome_session_stop_command, chrome_use_warning, classify_call_failure, eval_count,
    expect_outcome, extract_output, extract_snapshot_text, is_daemon_unavailable_code,
    is_daemon_unavailable_error, is_self_launched_browser_error, is_session_unresponsive_error,
    is_unreachable_tab_error, net_error_phrase, never_dispatched_error, parse_first,
    sanitize_timeout_message, self_launched_browser_error, self_launched_browser_note,
    truncated_output_error, unreachable_tab_message, unreached_browser_error,
    with_condition_timeout_note,
};
use crate::chrome::escape_js_single_quoted;
use crate::chrome::files::{
    OneFileTarget, judge_one_file_target, one_file_probe, single_file_input, upload_argv,
};
use crate::chrome::forms::{
    ExpectCond, ExtractGate, count_eval_js, expect_args, extract_gate, parse_count_op,
    parse_predicate, parse_state, validate_extract_getters, wait_args, wait_target,
};
use crate::chrome::spawn::{CliOutput, CliRun, CliSpawn, spawn_cli};
use crate::chrome::{ChromeCallClocks, CliRecovery, clocks};
use crate::util::{UnwrapPoison, is_http_url};
use crate::{Tool, Workspace};
use anyhow::Context;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

use std::sync::Arc;
use std::time::{Duration, Instant};
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
    /// Extract an element's rendered text by CSS selector or ref — the
    /// `get text` verb, whose own reader is `innerText || textContent`: rendered
    /// text, except for an element with no rendered text of its own, where the
    /// `textContent` fallback can include rendered-out `<script>`/`<style>` content.
    GetText { selector: String },
    /// Extract rendered text through an `innerText` eval, falling back to
    /// `get text` when the eval fails or answers empty. Kept for callers that
    /// ask for it; `get_text` is the preferred reader (it is the helper's own,
    /// not page-world JS the page can redefine).
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
    Find {
        /// Locator type: text (rendered text, case-sensitive substring),
        /// role (accessibility-tree role; `name` filters on the computed
        /// accessible name), label (the control's accessible label, which
        /// includes `aria-label`), placeholder/alt/title (those HTML
        /// attributes, matched as substrings — the snapshot shows accessible
        /// names, not these), testid,
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
        /// Require whole-string text match instead of a substring.
        exact: Option<bool>,
        /// Zero-based index for `by: "nth"`. Required when `by` is "nth".
        index: Option<u32>,
    },
    /// Capture a screenshot of the current page as a PNG on disk and inject
    /// it into the agent's conversation context as a native image part, so
    /// the model can visually inspect the rendered page. The output path is
    /// chosen by the tool (under the safe temp root), never by the model.
    Screenshot {},
    /// Attach local files to the page's file input — the input itself, the label
    /// that pairs with it, or a container holding it. Relative paths resolve against
    /// the workspace, and the result names the absolute file each one was read from.
    Upload {
        selector: String,
        /// A list of local paths; a single path string is taken as the one-element
        /// list models keep reaching for.
        #[serde(deserialize_with = "one_or_many_paths")]
        files: Vec<String>,
    },
}

/// Helper for `#[serde(default = "true_val")]` on boolean fields.
const fn true_val() -> bool {
    true
}

/// Deserialize a list of paths that a model may have written as one string.
fn one_or_many_paths<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(path) => vec![path],
        OneOrMany::Many(paths) => paths,
    })
}

/// Root session namespace of the interactive chrome tool. Every agent-run session
/// lives under `agent-tab-`: that keeps it out of the daemon orphan sweep, and
/// inside the protected-session set of the `mahbot chrome` frontend, which refuses
/// a bare `session stop` on one without `--force` — so the prefix must not change.
/// An `agent-tab-*` session belongs to the run that opened it: the run's own release
/// is its closer, and until that run is over the sweeps leave it alone — the
/// protected-session verb above and the run-end release's reclaim sweep (which closes
/// only a group no live run, no record and no resumable job claims) are the two
/// deliberate exceptions.
pub(crate) const AGENT_TAB_PREFIX: &str = "agent-tab-";

/// Logical name of the per-run default session.
const DEFAULT_TAB: &str = "default";

/// The Issues row an agent step's un-settled tab group is filed under — a named message, because the
/// Issues view dedupes on `(message, reason)` and a reworded sentence would file the same fact
/// twice. The reason is the failing leg's own ([`super::chrome_groups::settle`] answers a note).
const SETTLE_ISSUE_MESSAGE: &str = "the agent's browser tab group was not left named and collapsed";

/// The failure texts this module itself mints for a call that never left mahbot: no CLI
/// ([`ChromeTool::ensure_available`] and the dispatch's own context), a CLI that could not be
/// spawned, and the call cut off at mahbot's own bound. Consts so the failure classifiers
/// ([`ChromeTool::never_reached_the_owner_browser`] and [`ChromeTool::left_no_page`]) cannot drift
/// from what the dispatch fails with.
const CLI_UNAVAILABLE: &str = "chrome-use CLI is not available";
const CLI_SPAWN_FAILED: &str = "Failed to execute chrome-use CLI";
const CLI_BOUND_EXCEEDED: &str = "chrome-use did not answer within";

/// The one wording for a helper that cannot be used — absent or unusable — shared
/// by the availability gate and the dispatch's own lookup so one condition never
/// reads two ways.
const HELPER_UNUSABLE: &str = "the product's own copy of the helper is absent or unusable, \
     so nothing can reach the browser.";

/// The workspace a chrome-use dispatch records when the surface driving it has
/// none of its own: link enrichment summarizes a channel message, not a
/// workspace, so there is no path to name (see
/// [`crate::logs::StrayEvent::workspace`]).
const NO_WORKSPACE: &str = "";

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
/// scheme no agent id derives from is addressable by no run's release — the cost paid
/// by the runs in flight when per-boot random names were replaced. Its tabs are still
/// reclaimable where a live group title carries the name: the release sweep enumerates the
/// live `agent-tab-*` titles and derives the namespace from the name
/// ([`session_namespace`]), and what is lost for such a name is only the run it came from.
/// (Which title a group carries is chrome-use's grouping rule to decide — see
/// [`crate::tools::chrome_tabs`]' honest limits.)
pub(crate) fn run_session_namespace(agent_id: &str) -> String {
    let hex = crate::util::hex_string(&Sha256::digest(agent_id.as_bytes()));
    format!("{AGENT_TAB_PREFIX}{}-", &hex[..12])
}

/// The namespace a physical session name carries back — the inverse of
/// [`run_session_namespace`], for a name read out of the browser rather than derived
/// from an agent id. A name that is not one of this product's run sessions carries no
/// namespace and names only itself, so the one naming scheme stays in one place.
pub(crate) fn session_namespace(name: &str) -> &str {
    let Some(key) = name
        .strip_prefix(AGENT_TAB_PREFIX)
        .and_then(|rest| rest.split_once('-'))
        .map(|(key, _)| key)
    else {
        return name;
    };
    if key.is_empty() {
        name
    } else {
        // The separator found by `split_once` is the byte at this end.
        let namespace_len = AGENT_TAB_PREFIX.len() + key.len() + 1;
        &name[..namespace_len]
    }
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

/// One chrome action's output plus the leftover fact of every chrome-use call it
/// ran (the same pair [`ChromeTool::run_command`] returns): `true` when a call
/// finished while a process it left behind still held that call's output channel
/// ([`crate::chrome::spawn::CliOutput::leftover_pipes`]).
///
/// The child's exit status still decides the call and `output` is the answer; the
/// leftover is a NOTE on it ([`crate::chrome::contract::chrome_leftover_pipe_note`]),
/// never a failure — which is why it travels beside the output instead of turning
/// into an `Err`.
pub(super) struct ActionOutput<T> {
    pub(super) output: T,
    pub(super) leftover_pipes: bool,
}

/// Chrome tool for fetching content from web pages.
///
/// `tab` is optional — it names the page in this run's private session, and an empty or
/// missing value means the default page ([`TAB_ARG`]). Separate chrome sessions are isolated
/// via `--session`: every session an agent run dispatches is resolved to the run's private
/// namespace (`ChromeTool::resolve_session` plus `ChromeRunSessions`) and handed to that run's
/// release, so concurrent runs of different ids can never collide on or address each other's
/// sessions (see `run_session_namespace` for the same-id case and its transitional residue);
/// tool output keeps showing the logical tab name. Operations on the same tab are serialized
/// via a per-tab lock.
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
    /// yields exactly chrome-use's allowed charset). A name that already
    /// carries this namespace keeps it — but its tail is sanitized the same way
    /// a freshly minted one is, so every name this returns is `prefix` + a
    /// plain identifier and an echoed-back session name can never double-prefix
    /// into a different session. The shared non-agent instance (empty prefix)
    /// sanitizes the name the same way.
    fn resolve_session(&self, tab: &str) -> String {
        if let Some(tail) = tab.strip_prefix(&self.session_prefix) {
            // A name already carrying this namespace passes through — but its
            // tail is sanitized the same way a freshly minted one is, so every
            // name this returns is `prefix` + a plain identifier (the leftover
            // note's stop recipe is built from it).
            return format!(
                "{}{}",
                self.session_prefix,
                sanitize_filename_component(tail)
            );
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
    /// ARIA roles, and indentation), the eval returns plain rendered text — no
    /// markup, no hidden content, no `<script>`/`<style>` noise.
    ///
    /// Falls back to the `get text` verb (whose own reader is the helper's
    /// `innerText || textContent`) if the JavaScript eval fails or answers
    /// empty; the returned text then carries a note saying so, and that
    /// `textContent` side can surface rendered-out `<script>`/`<style>` text for
    /// an element with no rendered text of its own.
    ///
    /// The tab stays open for the run — the product closes the run's tabs itself
    /// when the run ends, retrying until the browser says they are gone; the
    /// per-tab lock is held for the full duration (navigate + extract) so
    /// concurrent callers targeting the same tab are serialized consistently.
    pub async fn fetch_page_text(&self, url: &str, tab: &str) -> anyhow::Result<String> {
        crate::util::http::validate_url(url)?;

        Self::require_browser().await?;

        // Lock is held for the entire navigate + extract sequence so
        // concurrent same-tab access doesn't race between navigation
        // and text extraction.
        let _guard = self.acquire_tab_lock(tab).await;
        // The elapsed time of the whole enrichment dispatch, carried onto the
        // durable record a leftover helper leaves behind below.
        let started = Instant::now();
        let mut leftover_pipes = false;

        let opened = self.run_command(&["open", url], tab, NO_WORKSPACE).await?;
        leftover_pipes |= opened.leftover_pipes;
        let opened = opened.output;

        // A real navigation that ends on the scratch `about:blank` or Chrome's
        // error page never loaded — fail loudly instead of returning empty
        // content.
        self.bail_on_failed_navigation(tab, url, &opened, NO_WORKSPACE)
            .await?;

        // Wait for network idle (best-effort — no hard error on timeout).
        if let Ok(call) = self
            .run_command(&["wait", "--load", "networkidle"], tab, NO_WORKSPACE)
            .await
        {
            leftover_pipes |= call.leftover_pipes;
        }

        // Extract clean visible text via innerText JS eval (not snapshot).
        let text = self.get_inner_text("body", tab, NO_WORKSPACE).await?;
        leftover_pipes |= text.leftover_pipes;

        // A helper this enrichment left behind gets its one durable record, like
        // the tool's own dispatch path. No note goes on the returned page text —
        // this call's result is the page, and the record is where the leftover
        // fact lives.
        if leftover_pipes {
            Self::record_leftover(&self.resolve_session(tab), NO_WORKSPACE, started.elapsed());
        }

        Ok(text.output)
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
        workspace: &str,
    ) -> anyhow::Result<()> {
        match crate::chrome::navigation_failure(response) {
            Some(crate::chrome::NavigationFailure::ErrorPage) => {
                // Best-effort: ask the error page which net error code Chrome
                // rendered (the same probe the CLI `open` uses); the generic
                // message is the fallback when nothing recognizable renders.
                let code = self
                    .run_command(&["eval", ERROR_PAGE_PROBE_JS], tab, workspace)
                    .await
                    .ok()
                    .and_then(|call| crate::chrome::error_page_code(&call.output));
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
            Some(crate::chrome::NavigationFailure::Blank) => {
                self.close_session(tab).await;
                anyhow::bail!(
                    "Navigation failed: the tab is still on a blank page after opening {url} — the \
                     page never loaded. This usually means the site is unreachable or blocks \
                     automated navigation, or the chrome-use extension relay is down."
                );
            }
            None => Ok(()),
        }
    }

    /// Fail with an actionable error when the chrome-use CLI is missing or the
    /// connection to the owner's real browser is not established, distinguishing
    /// the causes and never reporting a transient probe failure (spawn
    /// EAGAIN/EMFILE, timeout) as "not installed". One CLI probe; the readiness
    /// gate is cached.
    ///
    /// The gate ([`super::chrome_daemon::ensure_ready_for_actions`]) is what
    /// decides — a cached Down/Unknown verdict must never short-circuit it: the
    /// gate is the thing that RE-ESTABLISHES the connection (one bounded
    /// recovery pass, then a re-probe) before an action is dispatched, so
    /// bailing on the cached verdict would leave the connection down for as long
    /// as its cached readiness snapshot stands (10 s). When the gate refuses and
    /// the stored verdict classified a cause,
    /// [`super::chrome_daemon::refusal_message`] names what recovery is doing
    /// about it.
    ///
    /// The texts are audience-neutral — the page rasteriser shares this gate — so
    /// the chrome tool's own actions go through [`Self::require_browser`], which
    /// adds the agent's sentence.
    pub(super) async fn ensure_available() -> anyhow::Result<()> {
        match super::chrome_daemon::cli_probe().await {
            super::chrome_daemon::CliStatus::Available => {}
            super::chrome_daemon::CliStatus::Missing => {
                anyhow::bail!("{CLI_UNAVAILABLE}: {HELPER_UNUSABLE}");
            }
            super::chrome_daemon::CliStatus::Transient(failure) => {
                use super::chrome_daemon::CliProbeFailure;
                let cause = match failure {
                    CliProbeFailure::Spawn(reason) => format!(
                        "the helper could not be started ({reason}) — a host failure rather \
                         than a missing install; retry shortly."
                    ),
                    CliProbeFailure::BadVersion(status) => format!(
                        "the product's own copy of the helper is unusable — its `--version` \
                         check failed ({status})."
                    ),
                    CliProbeFailure::Timeout => "the helper probe timed out — the binary is \
                         present but unresponsive; retry shortly."
                        .to_string(),
                };
                anyhow::bail!(cause);
            }
        }
        if let Err(refusal) = super::chrome_daemon::ensure_ready_for_actions().await {
            anyhow::bail!("{}", super::chrome_daemon::refusal_message(&refusal));
        }
        Ok(())
    }

    /// [`Self::ensure_available`] as this tool's agent reads it: the gate's
    /// audience-neutral text plus [`super::chrome_daemon::REPORT_AS_IT_CAME`], added
    /// once here so no probe-failure branch can omit it.
    async fn require_browser() -> anyhow::Result<()> {
        Self::ensure_available()
            .await
            .map_err(|error| anyhow::anyhow!("{error}{}", super::chrome_daemon::REPORT_AS_IT_CAME))
    }

    /// Declared chrome-side condition-wait deadlines, forwarded to chrome-use
    /// via `--timeout` (see [`Self::build_args`]). chrome-use honours them in
    /// full: its own timeout surfaces at the declared deadline, so its own reason
    /// reaches the agent rather than a mahbot cut-off.
    const WAIT_DEADLINE: Duration = Duration::from_secs(10);
    const EXPECT_DEADLINE: Duration = Duration::from_secs(20);

    /// The recovery policy of every action this tool runs: each one is agent
    /// work, so chrome-use's own relay self-heal is always allowed. Used both
    /// where [`Self::call_clocks`] derives the clocks and where
    /// [`Self::spawn_call`] spawns the child, so the kill and the policy the
    /// call spawns with cannot drift apart.
    const ACTION_RECOVERY: CliRecovery = CliRecovery::Allowed;

    /// The two clocks of one dispatch, derived in ONE place
    /// ([`crate::chrome::clocks`]).
    ///
    /// `wait`/`expect` forward their declared deadline as `--timeout`, which
    /// chrome-use honours in full — the declared value IS their chrome-side
    /// clock. Every other verb forwards no deadline (`open`'s navigate included:
    /// chrome-use does not bound it by the forwarded env value), so the clock
    /// declared to chrome-use is
    /// [`CHROME_USE_DECLARED_BUDGET`](crate::chrome::CHROME_USE_DECLARED_BUDGET) — one
    /// kill margin under the tool's own client tolerance, so chrome-use always gives
    /// up with its own verdict first. Every action takes [`Self::ACTION_RECOVERY`]: the
    /// tool's actions are exactly the work chrome-use's self-heal is meant to
    /// protect.
    fn call_clocks(args: &[&str]) -> ChromeCallClocks {
        match args.first() {
            Some(&"wait") => clocks(Some(Self::WAIT_DEADLINE), Self::ACTION_RECOVERY),
            Some(&"expect") => clocks(Some(Self::EXPECT_DEADLINE), Self::ACTION_RECOVERY),
            _ => clocks(None, Self::ACTION_RECOVERY),
        }
    }

    /// Run one chrome-use command and parse the JSON response, then record the call on this run's
    /// per-agent tab record ([`Self::record_addressed`]) — the record counts what the browser
    /// really answered for, which only this call's outcome says, so the recording comes after it.
    async fn run_command(
        &self,
        args: &[&str],
        tab: &str,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<ChromeResponse>> {
        let outcome = self.dispatch_command(args, tab, workspace).await;
        self.record_addressed(tab, outcome.as_ref().err());
        outcome
    }

    /// Note one dispatched call on this run's tab record: the tab's recency — the agent's OWN
    /// addresses, never the browser's own last-activity stamp, which does not move while the agent
    /// works in a background tab — and, when the browser really answered for the page, its
    /// appearance. An appearance that puts the record over the cap names the tab to close; the
    /// close runs off this call's path, and the tab keeps counting as open until the browser
    /// confirms it is gone.
    ///
    /// A call the browser answered for is a real call even when it failed: it counts as an address
    /// to a tab the record already holds (the run was there, rather than neglecting it), as an
    /// appearance for a name the record does not hold (the session's page exists from the moment the
    /// relay answered it), and as a revival for a name the cap closed. A call that left no page at
    /// all — never got past the CLI, the daemon or the relay, went to a browser chrome-use launched
    /// itself, found no page of the session's to drive, or was cut off at mahbot's own bound with
    /// nothing to attribute it (see [`Self::left_no_page`]) — moves nothing: no tab, no recency, no
    /// revival, so no live tab is spent on a phantom and none is kept off the cap's list by one.
    pub(crate) fn record_addressed(&self, tab: &str, error: Option<&anyhow::Error>) {
        // The shared non-agent instance (link enrichment) keeps no record: it has no agent
        // run, and its own sessions are the product's ephemeral sweep family.
        if self.session_prefix.is_empty() {
            return;
        }
        let reached = error.is_none_or(|error| !Self::left_no_page(error));
        let Some(eviction) = super::chrome_tab_ledger::note_addressed(
            &self.session_prefix,
            self.logical_tab(tab),
            reached,
        ) else {
            return;
        };
        self.evict_over_cap_tab(&eviction, tab);
    }

    /// The record's own name for `tab`: the run's namespace is a transport detail of the session,
    /// not part of the logical name the agent uses, so a name that already carries it (an
    /// echoed-back physical name, which [`Self::resolve_session`] accepts) counts as the same tab
    /// rather than as a second one.
    ///
    /// A name the sanitizer had to change does not round-trip through this: its physical spelling
    /// carries the hasher suffix, and the two spellings are then two logical names — the price of
    /// keying the record by the names the agent uses.
    fn logical_tab<'a>(&'a self, tab: &'a str) -> &'a str {
        tab.strip_prefix(&self.session_prefix).unwrap_or(tab)
    }

    /// Whether a failed dispatch never reached the owner's browser: no CLI (or one that could not
    /// be spawned), a daemon or relay that never took the call ([`never_dispatched_error`]), or
    /// work that went to a browser chrome-use launched itself. Shared by the tab record's
    /// classifier ([`Self::left_no_page`]) and the step's group tail, which skips a step that
    /// provably never used a tab.
    fn never_reached_the_owner_browser(message: &str) -> bool {
        never_dispatched_error(message)
            || message.contains(CLI_UNAVAILABLE)
            || message.contains(CLI_SPAWN_FAILED)
            || is_self_launched_browser_error(message)
    }

    /// Whether a failed dispatch left the page it was asked for nowhere in the browser — the one
    /// class of failure that must not create a tab, refresh a tab's recency, or bring a name the
    /// cap closed back. Every failure that answers for a page — a missing element, a failed
    /// navigation, a chrome-use timeout on a page that is there — is a page the session really
    /// has, and counts.
    fn left_no_page(error: &anyhow::Error) -> bool {
        let message = error.to_string();
        // The browser's structural no-page classes, the texts this module mints, and the two
        // page-level classes only this record counts as no page: a call cut off at mahbot's own
        // bound, and a tab the session cannot drive. The group tail skips only the narrower
        // [`Self::never_reached_the_owner_browser`] class — a cut-off step may still have opened
        // its tab first.
        unreached_browser_error(&message)
            || Self::never_reached_the_owner_browser(&message)
            || message.contains(CLI_BOUND_EXCEEDED)
            || is_unreachable_tab_error(&message)
    }

    /// Close the tab the cap picked, off the agent's path: the call that went over the cap
    /// answers without waiting for a browser close (a close is several reads), and the tab
    /// keeps counting as open until the browser itself confirms it is gone — an unconfirmed
    /// close is simply attempted again at the next appearance over the cap.
    ///
    /// `current` is the tab this call addressed; the close's own reads run in its session —
    /// one the browser answered a moment ago — so they mint no page, which the victim's own
    /// session cannot promise (a record entry the browser has already lost would otherwise
    /// get a fresh scratch group).
    ///
    /// The pick is re-checked against the record right before the close (`take_eviction`): a run
    /// that has come back to the tab meanwhile keeps it, and the eviction is dropped rather than
    /// run under the call that just addressed it. A call already in flight when the close starts is
    /// knowingly not seen — it records its address only once it returns — so the close can land
    /// under it and that call answers with the unreachable-tab failure; the record still ends up
    /// right, and the run takes another name.
    fn evict_over_cap_tab(&self, eviction: &super::chrome_tab_ledger::Eviction, current: &str) {
        let namespace = self.session_prefix.clone();
        let session = self.resolve_session(&eviction.name);
        let read_session = self.resolve_session(current);
        let eviction = eviction.clone();
        tokio::spawn(async move {
            if !super::chrome_tab_ledger::take_eviction(&namespace, &eviction) {
                return;
            }
            let gone =
                super::chrome_release::evict_agent_tab(&namespace, &session, &read_session).await;
            super::chrome_tab_ledger::settle_eviction(&namespace, &eviction, gone);
        });
    }

    /// Whether the per-agent tab record holds this tab as closed by the cap: the page is
    /// gone, and addressing the name is an error until an `open` mints one again.
    fn tab_was_closed_by_cap(&self, tab: &str) -> bool {
        !self.session_prefix.is_empty()
            && super::chrome_tab_ledger::is_closed(&self.session_prefix, self.logical_tab(tab))
    }

    /// The refusal for a tab the cap closed: the name is gone until an `open` with an
    /// address mints a page under it again. Deliberately never phrased as a browser or
    /// extension problem — the tool's own record decides it, and the browser is not asked.
    fn cap_closed_tab_message(tab: &str) -> String {
        format!(
            "Tab \"{tab}\" no longer exists: the tab limit closed it — a run keeps at most {} \
             tabs open, and the one it has addressed least recently is closed when a new one \
             appears. Use another `tab` name, or `open` a URL under this name to create the tab \
             again.",
            super::chrome_tab_ledger::MAX_AGENT_TABS
        )
    }

    /// Run an chrome-use command in this run's session and parse the JSON
    /// response.
    ///
    /// The session is resolved here, at the single dispatch point, so a missing
    /// CLI never registers a pointless release — the run-end release queue only
    /// needs sessions that were actually dispatched; the call itself is
    /// [`Self::dispatch_call`].
    async fn dispatch_command(
        &self,
        args: &[&str],
        tab: &str,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<ChromeResponse>> {
        let cli = super::chrome_daemon::cli_path()
            .ok_or_else(|| anyhow::anyhow!("{CLI_UNAVAILABLE}: {HELPER_UNUSABLE}"))?;
        let session = self.resolve_session(tab);
        self.chrome_sessions.track(&session);

        let mut logged_args: Vec<&str> = args.to_vec();
        logged_args.extend(["--json", "--session", &session]);
        debug!("chrome-use args: {:?}", logged_args);

        Self::dispatch_call(&cli, &session, args, Self::call_clocks(args), workspace).await
    }

    /// One chrome-use call in `session`, read the one way every call in this tool
    /// is read: the child's own EXIT decides it ([`crate::chrome::spawn`]); a
    /// successful envelope that carries chrome-use's browser-replacement note is a
    /// failure because the work went to a browser chrome-use launched itself, not
    /// the owner's real one; a cut-off answer names itself; and a process the call
    /// left behind still holding its output channel is carried as a NOTE on that
    /// result, never as the failure — but it is recorded durably
    /// ([`Self::record_leftover`]) and named in the failure text
    /// ([`crate::chrome::contract::chrome_leftover_pipe_note`]).
    pub(super) async fn dispatch_call(
        cli: &Path,
        session: &str,
        args: &[&str],
        clocks: crate::chrome::ChromeCallClocks,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<ChromeResponse>> {
        // The elapsed time of this dispatch, carried on the durable rows it may
        // write (see [`Self::record_leftover`]).
        let started = Instant::now();
        let output = Self::spawn_call(cli, args, session, clocks).await?;
        let elapsed = started.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let leftover = output.leftover_pipes;
        // Every failure of this call leaves through `failed` below, so the
        // leftover this call holds is named and recorded for all of them alike —
        // and it never becomes the failure itself.
        let failed = match parse_run_output(
            args.first().copied(),
            output.truncated,
            output.status.success(),
            &stdout,
            &stderr,
        ) {
            ParsedRun::Ok(response) => {
                // Deliberately NOT the daemon-down path — the daemon and the
                // relay may be healthy, and what was lost is the connection to
                // the real browser — so health is left untouched (see
                // `self_launched_browser_error`). The call FAILS, so its failure
                // text is already its durable record; nothing else is written.
                if let Some(note) = self_launched_browser_note(&response) {
                    anyhow::anyhow!("{}", self_launched_browser_error(&note))
                } else {
                    return Ok(ActionOutput {
                        output: response,
                        leftover_pipes: leftover,
                    });
                }
            }
            ParsedRun::Unparseable => {
                if output.truncated {
                    anyhow::anyhow!("{}", truncated_output_error())
                } else {
                    anyhow::anyhow!("Failed to parse chrome-use JSON response")
                }
            }
            ParsedRun::Failed {
                error,
                code,
                retryable,
            } => {
                Self::failed_run_message(
                    error,
                    output.status,
                    code.as_deref(),
                    retryable,
                    args.first().copied().unwrap_or(""),
                    session,
                )
                .await
            }
        };
        if leftover {
            // A helper chrome-use started and left running is a fact of its own
            // beside the failure: recorded durably, and appended to the failure
            // text the agent reads (see [`chrome_leftover_pipe_note`]). The
            // failure keeps its own text and classification — a leftover never
            // decides the outcome.
            Self::record_leftover(session, workspace, elapsed);
            return Err(anyhow::anyhow!(
                "{}",
                crate::tools::with_note(&failed.to_string(), &chrome_leftover_pipe_note(session))
            ));
        }
        Err(failed)
    }

    /// Spawn one chrome-use call under `clocks` and collect its output. A spawn
    /// failure and a product-clock timeout are the call's own errors, and the
    /// timeout path probes the session first
    /// ([`super::chrome_daemon::health_after_call_timeout`] — with the RESOLVED
    /// session, which is what the call dispatched against): a wedged daemon is
    /// then reported as such and recovered, while a merely slow page keeps the
    /// product-bound "did not answer" text.
    async fn spawn_call(
        cli: &Path,
        args: &[&str],
        session: &str,
        clocks: crate::chrome::ChromeCallClocks,
    ) -> anyhow::Result<CliOutput> {
        match spawn_cli(CliSpawn {
            path: cli,
            args,
            session: Some(session),
            json: true,
            capture_stderr: true,
            timeout: clocks.kill,
            // A timed-out/cancelled call must not leave the chrome-use child
            // running its retry loop in the background.
            cancel_kills: true,
            input: None,
            chrome_side: clocks.chrome_side,
            recovery: Self::ACTION_RECOVERY,
        })
        .await
        {
            CliRun::Output(output) => Ok(output),
            CliRun::SpawnFailure => anyhow::bail!("{CLI_SPAWN_FAILED}"),
            CliRun::TimedOut => {
                // Distinguish "daemon down/wedged" (fail fast with daemon
                // guidance; wakes the watchdog) from "daemon healthy, that
                // call was just slow". The probe covers the wedge case a
                // status check cannot see — see health_after_call_timeout.
                if let Some(down_message) =
                    super::chrome_daemon::health_after_call_timeout(session).await
                {
                    anyhow::bail!("{down_message}");
                }
                anyhow::bail!(
                    "{CLI_BOUND_EXCEEDED} {}s for `{}` — that is mahbot's own bound, \
                     which rides above the {}s deadline chrome-use itself was working to (plus its \
                     relay self-heal window), so the call was aborted before chrome-use could \
                     report its own reason. The daemon looked healthy, so this is usually a slow \
                     page or a wedged CLI; retry once before trying another approach.",
                    clocks.kill.as_secs(),
                    args.join(" "),
                    clocks.chrome_side.as_secs()
                );
            }
        }
    }

    /// The durable record, written into the log store (shown on Logs → Issues),
    /// of the helper a FINISHED chrome-use call left behind: chrome-use's own
    /// session daemon, still holding that call's output channel — started
    /// deliberately, left running, and stoppable by the recipe
    /// ([`chrome_session_stop_command`]) the record carries as its detail, so the
    /// agent's note ([`chrome_leftover_pipe_note`]) and the record cannot drift.
    /// The row says what happened and who ran it, taken from the calling agent's
    /// own attribution ([`crate::agent::tool_record_attribution`]).
    ///
    /// One record per reported fact, never duplicated within a call: the failure
    /// branches of [`Self::run_command`] write it for their call, and the two
    /// successful paths that keep no failure text — [`Tool::execute`] for the
    /// action's own dispatch and [`Self::fetch_page_text`] for a link-enrichment
    /// fetch — write it once for the call whose chrome-use calls held the
    /// leftover. `elapsed` is the time the record's call took — that chrome-use
    /// call's dispatch for a failure record, the whole dispatch for the other two.
    pub(super) fn record_leftover(session: &str, workspace: &str, elapsed: Duration) {
        crate::logs::record_stray_event(&crate::logs::StrayEvent {
            source: crate::logs::STRAY_SOURCE_CHROME_TOOL,
            message: "chrome-use exited while a helper it had started still held the call's output \
                      channel, so no more of its output could be collected",
            recipe: &chrome_session_stop_command(session),
            scope: session,
            session,
            workspace,
            duration_ms: Some(crate::util::millis_i64(elapsed)),
        });
    }

    /// The failure text of a `ParsedRun::Failed` call: chrome-use's own error
    /// (or the exit status when the envelope carried none), the retry hint, the
    /// verb's condition-timeout note, and — when it fires — the daemon-down
    /// guidance that REPLACES the classification.
    async fn failed_run_message(
        error: String,
        exit: std::process::ExitStatus,
        code: Option<&str>,
        retryable: Option<bool>,
        verb: &str,
        session: &str,
    ) -> anyhow::Error {
        let error_msg = if error.is_empty() {
            format!("chrome-use exited with code {exit}")
        } else {
            enhance_chrome_error(error)
        };
        // Classify once: strip chrome-use's stale-relay hint from Timeout
        // messages (mahbot's surface has no `connect` verb; Environment
        // diagnostics pass through untouched), then append the action's
        // remediation note from the same kind.
        let kind = classify_call_failure(code, &error_msg);
        let error_msg = sanitize_timeout_message(kind, &error_msg);
        let mut error_msg = with_retry_hint(error_msg, retryable);
        with_condition_timeout_note(verb, kind, &mut error_msg);
        // The daemon-down guidance REPLACES the classification when it fires, so
        // it is that text a leftover note follows.
        match Self::fail_fast_if_daemon_down(&error_msg, code, session).await {
            Err(guidance) => guidance,
            Ok(()) => anyhow::anyhow!("{}: {error_msg}", kind.as_str()),
        }
    }

    /// If an error carries the daemon-unavailable signature or envelope code,
    /// mark the daemon unhealthy (wakes the auto-recovery watchdog) and return
    /// the actionable guidance immediately — the CLI already retried
    /// internally, so adding more retries would only burn more time.
    /// Unreachable-tab errors are their own state: the daemon and relay are up,
    /// only the session's tab is orphaned — fail fast with the unreachable-tab
    /// guidance (the tab is the product's to close, at run end or on recovery,
    /// not the agent's to touch) and leave health untouched (recovery cannot fix
    /// a Chrome-side orphan, and hiding the daemon would block other sessions
    /// until the next healthy probe).
    ///
    /// A SESSION-unresponsive signature is recovered instead of only marked
    /// down: [`super::chrome_daemon::recover_unresponsive_session`] records the
    /// wedge (which wakes the watchdog, so the background path recovers too) and
    /// stops `session`'s daemon so the next call gets a clean one — the error
    /// text says what was done.
    async fn fail_fast_if_daemon_down(
        error: &str,
        code: Option<&str>,
        session: &str,
    ) -> anyhow::Result<()> {
        if is_unreachable_tab_error(error) {
            anyhow::bail!("{}", unreachable_tab_message(error));
        }
        if is_daemon_unavailable_error(error) || is_daemon_unavailable_code(code) {
            if is_session_unresponsive_error(error) {
                let recovery = super::chrome_daemon::recover_unresponsive_session(session).await;
                anyhow::bail!(
                    "{} {}",
                    super::chrome_daemon::daemon_down_message(),
                    recovery.summary()
                );
            }
            super::chrome_daemon::note_unhealthy(error);
            anyhow::bail!("{}", super::chrome_daemon::daemon_down_message());
        }
        Ok(())
    }

    /// Extract visible rendered text via `innerText`, falling back to the
    /// `get text` verb when eval fails or returns empty. The fallback note says
    /// which reader answered, since `get text` is the helper's own and can
    /// differ from the page-world eval. The leftover flag is the same record
    /// every chrome-use call carries
    /// ([`ActionOutput::leftover_pipes`]), over all the calls this action ran.
    async fn get_inner_text(
        &self,
        selector: &str,
        tab: &str,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<String>> {
        const FALLBACK_NOTE: &str = "(answered by the get text fallback, not the innerText eval)";

        let js = inner_text_eval_js(selector);
        let mut leftover_pipes = false;
        if let Ok(call) = self.run_command(&["eval", &js], tab, workspace).await {
            leftover_pipes |= call.leftover_pipes;
            if let Some(data) = call.output.data.as_ref()
                && let Some(text) = extract_snapshot_text(data)
                && !text.trim().is_empty()
            {
                return Ok(ActionOutput {
                    output: text,
                    leftover_pipes,
                });
            }
        }

        let call = self
            .run_command(&["get", "text", selector], tab, workspace)
            .await?;
        leftover_pipes |= call.leftover_pipes;
        let mut text = call
            .output
            .data
            .as_ref()
            .and_then(extract_snapshot_text)
            .unwrap_or_default();
        if !text.is_empty() {
            text.push('\n');
            text.push_str(FALLBACK_NOTE);
        }
        Ok(ActionOutput {
            output: text,
            leftover_pipes,
        })
    }

    /// The chrome-use CLI takes a different argument shape per action — this
    /// builds the correct argument list for each action.
    #[expect(clippy::too_many_lines)] // the per-action argv shapes are one cohesive dispatch
    fn build_args(action: &ChromeAction) -> anyhow::Result<Vec<String>> {
        match action {
            ChromeAction::Open { url } => {
                crate::util::http::validate_url(url)?;
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
            ChromeAction::Upload { .. } => {
                anyhow::bail!("Upload is handled in execute(), not build_args")
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
        // the CLI help), so it may differ from the order the hand-built schema
        // used; the set is the registry's `tool` entries, and nothing else.
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
                    "description": TAB_ARG
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, ws: &Workspace, args: Value) -> anyhow::Result<String> {
        let (tab, action, normalized_notes) = Self::normalize_call(&args)?;
        for note in &normalized_notes {
            super::log_normalization_note("chrome", note);
        }

        debug!(tab, action = ?action, "chrome action");

        // A tab the per-agent cap closed is gone: only an `open` with an address mints a
        // page under that name again. Decided from the tool's own record, before the
        // browser is probed at all, so the refusal can never be replaced by an
        // "unreachable browser" error (see [`Self::cap_closed_tab_message`]).
        if self.tab_was_closed_by_cap(&tab) && !matches!(action, ChromeAction::Open { .. }) {
            anyhow::bail!("{}", Self::cap_closed_tab_message(&tab));
        }

        Self::require_browser().await?;

        Self::validate_find(&action)?;

        // Get or create a per-tab lock — only serializes operations on the
        // same tab. Different tabs run fully concurrently.
        let _guard = self.acquire_tab_lock(&tab).await;

        // The workspace is carried onto the durable rows below.
        let workspace = ws.as_path().to_string_lossy().into_owned();

        let outcome = self.run_step(&tab, action, &workspace).await;
        // The cosmetic tail rides the step, inside the same per-tab lock: the
        // step's own result is already decided and is never changed by it.
        self.settle_step_group(&tab, outcome.as_ref().err()).await;
        outcome
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
    /// One browser step with the per-tab lock already held: the action's own
    /// dispatch and its output formatting. The step's cosmetic tail is the
    /// caller's ([`Self::settle_step_group`]), so it runs on a failure too.
    async fn run_step(
        &self,
        tab: &str,
        action: ChromeAction,
        workspace: &str,
    ) -> anyhow::Result<String> {
        // The elapsed time of the chrome-use dispatch this action runs (it may be
        // several calls — `run_command` measures each call of its own).
        let started = Instant::now();

        // Each action path reports whether the chrome-use call it drove exited
        // while a process it left behind still held the call's output channel: the
        // call finished either way, and that leftover is a note on the result
        // below, never an error (see [`chrome_leftover_pipe_note`]).
        let ActionOutput {
            output: body,
            leftover_pipes,
        } = if let ChromeAction::GetInnerText { selector } = &action {
            let ActionOutput {
                output,
                leftover_pipes,
            } = self.get_inner_text(selector, tab, workspace).await?;
            let body = if output.is_empty() {
                format!("[Tab: {tab}] (no output)")
            } else {
                format!("[Tab: {tab}] {output}")
            };
            ActionOutput {
                output: body,
                leftover_pipes,
            }
        } else if let ChromeAction::Screenshot { .. } = &action {
            let ActionOutput {
                output,
                leftover_pipes,
            } = self.capture_screenshot(tab, workspace).await?;
            ActionOutput {
                output,
                leftover_pipes,
            }
        } else if let ChromeAction::Upload { selector, files } = &action {
            let ActionOutput {
                output,
                leftover_pipes,
            } = self.upload(selector, files, tab, workspace).await?;
            ActionOutput {
                output,
                leftover_pipes,
            }
        } else if let ChromeAction::Extract { schema, limit } = &action {
            let (schema, schema_note) = Self::normalize_extract_schema(schema)?;
            if let Some(note) = &schema_note {
                super::log_normalization_note("chrome", note);
            }
            self.run_extract(&schema, *limit, tab, workspace).await?
        } else {
            let ActionOutput {
                output: (response, snapshot),
                leftover_pipes,
            } = self.run_action(&action, tab, workspace).await?;
            let body = Self::format_action_output(&action, tab, response, &snapshot);
            ActionOutput {
                output: body,
                leftover_pipes,
            }
        };
        if leftover_pipes {
            let session = self.resolve_session(tab);
            // The durable record of the helper chrome-use left behind, in the
            // same wording the agent's note carries (see [`Self::record_leftover`]).
            Self::record_leftover(&session, workspace, started.elapsed());
            return Ok(crate::tools::with_note(
                &body,
                &chrome_leftover_pipe_note(&session),
            ));
        }
        Ok(body)
    }

    /// Leave this step's tab group as the agent's own — named `✦ agents` and collapsed — see
    /// [`super::chrome_groups`].
    /// Cosmetic and best-effort: a tail that did not work is filed once as the product's own issue
    /// (the Issues view dedupes it, so a host that cannot have the group does not report on every
    /// step), and it can never turn the step into a failed one. Skipped when the step provably never
    /// reached the owner's browser ([`Self::never_reached_the_owner_browser`]), for the shared
    /// non-agent instance (link enrichment's sessions are out of scope), and when the CLI binary
    /// cannot be resolved at all.
    async fn settle_step_group(&self, tab: &str, step: Option<&anyhow::Error>) {
        if self.session_prefix.is_empty()
            || step.is_some_and(|error| Self::never_reached_the_owner_browser(&error.to_string()))
        {
            return;
        }
        let Some(cli) = super::chrome_daemon::cli_path() else {
            return;
        };
        let session = self.resolve_session(tab);
        if let Some(note) = super::chrome_groups::settle(&cli, &session).await {
            crate::logs::record_issue_once(
                SETTLE_ISSUE_MESSAGE,
                note.reason,
                super::chrome_release::ISSUE_TARGET,
                json!({ "detail": note.detail, "session": session }),
            )
            .await;
        }
    }

    /// Resolve the tab for a call. An explicit non-empty tab passes through as the
    /// logical name; missing/empty falls back to the per-run default logical session,
    /// which [`ChromeTool::resolve_session`] maps into this run's namespace (a function
    /// of the agent id — see [`run_session_namespace`]). Defaulting is reported as a
    /// normalization note (the note carries the logical name only).
    fn normalize_tab(args: &Value) -> (String, Option<String>) {
        if let Some(tab) = super::get_opt_str(args, "tab").filter(|s| !s.is_empty()) {
            (tab.to_string(), None)
        } else {
            let note = format!("missing/empty tab defaulted to \"{DEFAULT_TAB}\"");
            (DEFAULT_TAB.to_string(), Some(note))
        }
    }

    /// Capture a screenshot of the current tab to a PNG under the safe temp
    /// root, record its path, and return a textual result describing it (plus the
    /// same leftover record every chrome-use call carries). The per-tab lock is
    /// already held by the caller.
    async fn capture_screenshot(
        &self,
        tab: &str,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<String>> {
        let path = Self::screenshot_output_path(tab)?;
        let path_str = path.to_string_lossy().into_owned();
        let str_args = ["screenshot", path_str.as_str()];
        let call = self.run_command(&str_args, tab, workspace).await?;
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
        let (width, height) = crate::chrome::files::validate_png(&path)?;
        *self.last_screenshot.lock().unwrap_poison() = Some(path_str.clone());
        let mut output = format!(
            "[Tab: {tab}] Captured a chrome screenshot: {path_str} ({width}x{height}). \
             [IMAGE:{path_str}]"
        );
        // chrome-use warns when the pixels did not come from the page the caller
        // thinks it captured; the file is still a picture, so the capture stands
        // and the warning is repeated rather than swallowed (as the CLI's does).
        if let Some(warning) = chrome_use_warning(&call.output) {
            use std::fmt::Write as _;
            let _ = write!(
                output,
                "\n\n⚠ chrome-use warning: {} — the picture was still written; check it \
                 shows the page you meant",
                crate::chrome::contract::warning_text(&warning)
            );
        }
        Ok(ActionOutput {
            output,
            leftover_pipes: call.leftover_pipes,
        })
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

    /// Attach local files to the page's file input — the input itself, a hidden one
    /// behind a styled control, the label that pairs with it, or a container holding
    /// it. Every path resolves against the workspace and is handed to
    /// chrome-use as the absolute file the result then names.
    ///
    /// Delivery is reported in chrome-use's own terms: its `uploaded`/`attached`
    /// counts, and its `warning` verbatim when it could not confirm the input still
    /// holds the files (a React-style control consumed or replaced them). That
    /// warning is a degraded SUCCESS here — unlike the text-input warnings, the
    /// files did reach the page — and nothing here can claim the page's own
    /// validation accepted them.
    async fn upload(
        &self,
        selector: &str,
        files: &[String],
        tab: &str,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<String>> {
        if files.is_empty() {
            anyhow::bail!("Not delivered — no local files were named for the attach");
        }
        let mut resolved = Vec::with_capacity(files.len());
        for raw in files {
            resolved.push(
                crate::chrome::files::resolve_local_file(raw, Path::new(workspace))
                    .map_err(|reason| anyhow::anyhow!("Not delivered — {reason}"))?,
            );
        }
        let mut leftover_pipes = false;
        // Several files at a one-file input is a FileList no real selection can
        // produce; the files module owns the question and its verdict, this only asks
        // the page. A probe that cannot answer never refuses the attach — the answer
        // says the check was skipped instead.
        let mut unchecked = false;
        if let Some(probe) = one_file_probe(selector, resolved.len()) {
            match self.run_command(&["eval", &probe], tab, workspace).await {
                Ok(call) => {
                    leftover_pipes |= call.leftover_pipes;
                    match judge_one_file_target(selector, single_file_input(&call.output)) {
                        OneFileTarget::Refused(reason) => {
                            anyhow::bail!("Not delivered — {reason}");
                        }
                        OneFileTarget::Unjudged => unchecked = true,
                        OneFileTarget::Proceed => {}
                    }
                }
                Err(error) => {
                    debug!("upload one-file probe did not run: {error}");
                    unchecked = true;
                }
            }
        }
        let argv = upload_argv(selector, &resolved);
        let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
        let call = match self.run_command(&refs, tab, workspace).await {
            Ok(call) => call,
            // A step the product's own clock ended — including the daemon-health
            // verdict that probe runs after such a kill — never learned what reached
            // the page, so its verdict is the unconfirmed delivery it is, and a plain
            // retry is not the next thing to do: a queued or half-streamed attach may
            // still land.
            Err(error) if Self::upload_delivery_unknown(&error) => {
                return Err(anyhow::anyhow!(
                    "Delivery is not confirmed — treat the files as not attached, and check the \
                     page before attaching them again: {error}"
                ));
            }
            // Any other failure may have streamed part of a file before giving up,
            // so it never claims the files are on the input either.
            Err(error) => return Err(anyhow::anyhow!("Attach did not complete — {error}")),
        };
        leftover_pipes |= call.leftover_pipes;
        // A browser answer of zero uploaded files is not delivery, so the call fails
        // as the CLI's does — the message keeps the counts, the files and the
        // browser's own words.
        if crate::chrome::files::upload_counts(&call.output).0 == Some(0) {
            return Err(anyhow::anyhow!(
                "Not delivered — {}",
                Self::upload_output(tab, selector, &resolved, &call.output, unchecked)
            ));
        }
        Ok(ActionOutput {
            output: Self::upload_output(tab, selector, &resolved, &call.output, unchecked),
            leftover_pipes,
        })
    }

    /// Whether a failed attach's cause leaves delivery unknown: the product's own
    /// clock ended the step ([`CLI_BOUND_EXCEEDED`]), or the daemon-health probe
    /// that rides above such a kill reported the helper unreachable.
    fn upload_delivery_unknown(error: &anyhow::Error) -> bool {
        let text = error.to_string();
        text.contains(CLI_BOUND_EXCEEDED) || is_daemon_unavailable_error(&text)
    }

    /// The `upload` answer: chrome-use's own counts, the absolute file each path was
    /// read from, and its warning. A warning marks a delivered attach the page may
    /// have consumed; it never claims the page's own validation accepted the files.
    /// A zero upload reads as nothing reaching the page and fails the call.
    fn upload_output(
        tab: &str,
        selector: &str,
        resolved: &[PathBuf],
        response: &ChromeResponse,
        unchecked: bool,
    ) -> String {
        use std::fmt::Write as _;
        let (uploaded, attached) = crate::chrome::files::upload_counts(response);
        let paths = resolved
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        // Never invent a count chrome-use did not state: an honest "the N it was
        // given" beats a delivered claim the browser never made. A stated zero is
        // the opposite of delivery and never reads as one.
        let mut output = match uploaded {
            Some(0) => format!(
                "[Tab: {tab}] Nothing reached the page for \"{selector}\": {paths} — the browser \
                 side reported 0 of {} file(s) uploaded",
                resolved.len()
            ),
            Some(uploaded) => format!(
                "[Tab: {tab}] Delivered {uploaded} of {} file(s) to \"{selector}\": {paths}",
                resolved.len()
            ),
            None => format!(
                "[Tab: {tab}] Delivered the {} file(s) it was given for \"{selector}\": {paths}",
                resolved.len()
            ),
        };
        if let Some(warning) = chrome_use_warning(response) {
            let text = crate::chrome::contract::warning_text(&warning);
            // A warning rides a delivered attach; with nothing uploaded it is just
            // the browser side's own words, never a delivery claim.
            if uploaded == Some(0) {
                let _ = write!(output, "\n\n⚠ chrome-use warning: {text}");
            } else {
                let _ = write!(
                    output,
                    "\n\n⚠ chrome-use warning: {text} — the files did reach the page, so this is \
                     not a failed attach; it is not a claim that the page's own validation \
                     accepted them"
                );
            }
        } else if let (Some(uploaded), Some(attached)) = (uploaded, attached) {
            // Only a real delivery can be a kept claim.
            if uploaded > 0 {
                if attached == uploaded {
                    output.push_str(" — the file input still holds them");
                } else {
                    // The input itself disagrees with what was sent (a transport
                    // that kept one file, or a page that added or dropped one):
                    // report what it holds instead of passing the sent count off as
                    // delivered.
                    let _ = write!(output, " — the file input holds {attached} of them");
                }
            }
        }
        if unchecked {
            let _ = write!(
                output,
                "\n\nNote: {}",
                crate::chrome::files::unchecked_several_files_note()
            );
        }
        output
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
        // The note names the reshaping, never the reshaped action: a `fill` carries text the
        // owner typed, and a note is only ever logged.
        normalized_notes.extend(normalized_note);

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
                "Invalid chrome action arguments{hint}. Expected action to be {EXPECTED_ACTION_SHAPE}. \
                 {TAB_ARG}. Serde error: {e}"
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
            // mahbot's own allowlist: only the sub-actions this tool models are
            // accepted, so a helper-side one it does not offer is refused here.
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
    /// and the compact auto-snapshot. Returns the RAW response, the snapshot
    /// text, and the leftover fact of every chrome-use step the action ran — no
    /// LLM-facing framing (that lives in [`Self::format_action_output`]; the
    /// leftover is a note on the result, see [`chrome_leftover_pipe_note`]).
    async fn run_action(
        &self,
        action: &ChromeAction,
        tab: &str,
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<(ChromeResponse, String)>> {
        let cli_args = Self::build_args(action)?;
        let str_args: Vec<&str> = cli_args.iter().map(String::as_str).collect();
        let call = self.run_command(&str_args, tab, workspace).await?;
        let mut leftover_pipes = call.leftover_pipes;
        let response = call.output;

        // A real navigation that ends on the scratch `about:blank` or Chrome's
        // error page never loaded — fail loudly (closing the blank-page tab
        // best-effort) instead of reporting success with no content.
        if let ChromeAction::Open { url } = action {
            self.bail_on_failed_navigation(tab, url, &response, workspace)
                .await?;
        }

        // After open, wait for network idle, then auto-snapshot
        // so the LLM sees page content immediately. Both are composite
        // sub-steps of this one tool call, but each runs as a chrome-use call of
        // its own under [`Self::call_clocks`], so — unlike the CLI's
        // product-bounded `open` operation — no sub-step here is cut off below
        // chrome-use's own clock.
        let snapshot = if matches!(action, ChromeAction::Open { .. }) {
            let wait_args = ["wait", "--load", "networkidle"];
            if let Ok(call) = self.run_command(&wait_args, tab, workspace).await {
                leftover_pipes |= call.leftover_pipes;
            }

            // Compact snapshot to return page content; when it comes back
            // empty (canvas/PDF/SPA shells, capture failure) fall back to
            // visible text so open still yields something readable.
            let snap = self
                .run_command(&["snapshot", "-c"], tab, workspace)
                .await
                .ok()
                .and_then(|call| {
                    leftover_pipes |= call.leftover_pipes;
                    call.output.data.as_ref().and_then(extract_snapshot_text)
                })
                .unwrap_or_default();
            if snap.trim().is_empty() {
                self.get_inner_text("body", tab, workspace)
                    .await
                    .map(|out| {
                        leftover_pipes |= out.leftover_pipes;
                        out.output
                    })
                    .unwrap_or_default()
            } else {
                snap
            }
        } else {
            String::new()
        };

        Ok(ActionOutput {
            output: (response, snapshot),
            leftover_pipes,
        })
    }

    /// Shape the raw [`ChromeResponse`] into the LLM-facing text: the
    /// per-action data extraction and the `[Tab: {tab}]` framing.
    fn format_action_output(
        action: &ChromeAction,
        tab: &str,
        response: ChromeResponse,
        snapshot: &str,
    ) -> String {
        use std::fmt::Write as _;
        // This shared path renders the text-input warnings only (`type` read-back
        // mismatch, `press` with no key listeners / provably nowhere); `upload` and
        // `screenshot` carry their own warnings in their own branches. The
        // browser-replacement note is never one of these: [`Self::run_command`]
        // already failed the call for it before any response reaches here.
        let warning = if matches!(
            action,
            ChromeAction::Fill { .. } | ChromeAction::Type { .. } | ChromeAction::Press { .. }
        ) {
            chrome_use_warning(&response)
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
            let _ = write!(
                output,
                "\n\n⚠ chrome-use warning: {} — treat the action as NOT applied and choose another approach",
                crate::chrome::contract::warning_text(&warning)
            );
        }

        output
    }

    /// Accept the model-facing extract schema: a JSON object, or a string
    /// holding JSON text (models frequently double-encode). The object must
    /// follow chrome-use's grammar — an optional `"rows"` selector string and a
    /// `"fields"` object (field name → CSS selector or `{sel, get, all}`). An
    /// absent `"fields"` (a flat schema with per-field keys at the top level, or
    /// a rows-only schema) is synthesized from the other keys and the correction
    /// is reported as a note. Array-shaped objects (`{items: [...]}`) are
    /// rejected — wrapping them into `fields` would hand chrome-use a malformed
    /// schema.
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
        workspace: &str,
    ) -> anyhow::Result<ActionOutput<String>> {
        let mut leftover_pipes = false;
        let rows_sel = schema.get("rows").and_then(Value::as_str);
        if let Some(sel) = rows_sel {
            let js = count_eval_js(sel);
            let call = self.run_command(&["eval", &js], tab, workspace).await?;
            leftover_pipes |= call.leftover_pipes;
            let count = eval_count(&call.output).ok_or_else(|| {
                anyhow::anyhow!(
                    "count eval for the rows selector '{sel}' returned a non-numeric result"
                )
            });
            match extract_gate(count) {
                ExtractGate::Empty => {
                    return Ok(ActionOutput {
                        output: serde_json::to_string_pretty(&extract_output(
                            &Value::Array(vec![]),
                            limit,
                        ))
                        .context("serialize extract output")?,
                        leftover_pipes,
                    });
                }
                ExtractGate::Proceed => {}
                ExtractGate::Fail(e) => return Err(e),
            }
        }
        let schema_str = serde_json::to_string(schema)?;
        let call = self
            .run_command(&["extract", "--schema", &schema_str], tab, workspace)
            .await?;
        leftover_pipes |= call.leftover_pipes;
        let Some(data) = call.output.data.as_ref() else {
            anyhow::bail!("extract returned no data");
        };
        Ok(ActionOutput {
            output: serde_json::to_string_pretty(&extract_output(data, limit))
                .context("serialize extract output")?,
            leftover_pipes,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────

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
    /// An answer that could not be parsed: stdout was not a chrome-use envelope,
    /// or the product cut it off
    /// ([`crate::chrome::spawn::CliOutput::truncated`]) — the caller names the
    /// cut-off case (see [`truncated_output_error`]).
    Unparseable,
}

/// Parse one chrome-use CLI output. `expect` is the one chrome-use action
/// whose `--json` envelope can succeed on a non-zero exit — a failed
/// assertion arrives as `success:true` with exit 1, the verdict riding in
/// `data.pass` — so envelope-success is trusted over the exit code there and
/// nowhere else. `truncated` is the product's own flag: an answer the product
/// cut off names itself rather than reading as chrome-use's malformed output,
/// whatever the exit status. The parse is the shared tolerant one
/// ([`crate::chrome::contract::parse_first`]), so trailing bytes a leftover
/// process wrote after the envelope are ignored. Pure so tests pin the mapping.
fn parse_run_output(
    action: Option<&str>,
    truncated: bool,
    status_success: bool,
    stdout: &str,
    stderr: &str,
) -> ParsedRun {
    let parsed: Option<ChromeResponse> = parse_first::<ChromeResponse>(stdout.as_bytes());
    let Some(resp) = parsed else {
        return if status_success || truncated {
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
///
/// The navigation hint exists because chrome-use's own `navigation_incomplete`
/// answer already says to re-check with `get url`/`snapshot` and not to repeat the
/// open, and recommends its CLI flag (`open <url> --wait-until none`), which this
/// tool does not offer: the agent gets the wait it can actually make instead.
fn enhance_chrome_error(msg: String) -> String {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("unknown ref")
        || lower.contains("node with given id does not belong to the document")
    {
        format!(
            "{msg}. Hint: refs become stale after any navigation or DOM change. \
             Take a fresh snapshot before using refs again."
        )
    } else if lower.contains("navigation_incomplete") {
        format!(
            "{msg}. Hint: this tool has no `--wait-until` — wait for what you need with `wait` \
             (`url`, `text` or a CSS selector) or `expect` rather than opening the URL again."
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
    "upload",
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
    {\"screenshot\":{}}, {\"upload\":{\"selector\":\"...\",\"files\":[\"...\"]}}, \
    {\"wait\":{\"selector\":\"...\"}}, \
    {\"expect\":{\"condition\":\"visible|hidden|present|count|text|value|attr|url\",\
    \"selector\":\"...\",\"count\":int,\"op\":\"==\",\"predicate\":\"equals|contains|matches\",\"name\":\"...\",\"expected\":\"...\"}}, \
    {\"extract\":{\"schema\":{...},\"limit\":int}}";

/// The one statement of what `tab` is, carried by the argument schema, both corrective errors
/// and [`ChromeTool`]'s doc: optional, and an empty or missing value means the default page.
const TAB_ARG: &str = "`tab` is optional — it names the page in this run's private session, and an \
                       empty or missing value means the default page";

/// Corrective error for an unrecoverable action shape, listing the exact
/// expected form instead of raw serde text.
fn corrective_action_error(received: &Value) -> String {
    format!(
        "Invalid chrome action arguments. Expected action to be {EXPECTED_ACTION_SHAPE}. {TAB_ARG}. \
         Received: {received}"
    )
}

/// Normalize a model-supplied chrome `action` value into the canonical
/// `{"variant": {...}}` tagged form that `ChromeAction` deserializes from.
///
/// Recoverable shapes are mapped to their canonical equivalent and a note
/// describing the correction is returned, so a silently reshaped call stays
/// diagnosable. Unrecoverable shapes produce a corrective error.
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

    // ── validate_find: refused actions ────────────────────────────────────

    /// focus/type/uncheck are mahbot's own refusal (the tool does not model
    /// their argument shapes), not the helper's: validate_find rejects them up
    /// front.
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

    // ── call_clocks policy ───────────────────────────────────────────────

    #[test]
    fn call_clocks_declare_the_verb_clock_and_kill_above_it() {
        // The kill's arithmetic has ONE home and one pinning test
        // ([`crate::chrome::clocks`] / [`crate::chrome::kill_bound`]), so these
        // expectations are expressed through the derivation itself: what this
        // test pins is the tool's own policy — WHICH clock each verb declares.
        let call_clocks = ChromeTool::call_clocks;
        // wait/expect forward their declared deadline as `--timeout`, so it IS
        // the chrome-side clock.
        assert_eq!(
            call_clocks(&["wait", "--load", "networkidle"]),
            crate::chrome::clocks(Some(ChromeTool::WAIT_DEADLINE), CliRecovery::Allowed)
        );
        assert_eq!(
            call_clocks(&["expect", "visible", "#x"]),
            crate::chrome::clocks(Some(ChromeTool::EXPECT_DEADLINE), CliRecovery::Allowed)
        );
        // Every other verb — `open` included, whose navigate chrome-use does not
        // bound by the forwarded env value — has no command-level deadline, so
        // the clock mahbot declares to chrome-use IS its clock: the call runs to that
        // clock and chrome-use reports first, which is what the product's kill
        // must always allow.
        for args in [
            &["open", "https://example.com"][..],
            &["click", "@e1"][..],
            &["eval", "1+1"][..],
            &["snapshot", "-c"][..],
            &["extract", "--schema", "{}"][..],
        ] {
            assert_eq!(
                call_clocks(args),
                crate::chrome::clocks(None, CliRecovery::Allowed),
                "args: {args:?}"
            );
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

    fn assert_rejects(action: serde_json::Value, args: &serde_json::Value) {
        let err = normalize_action(action, args).unwrap_err();
        assert!(
            err.contains("Expected action to be"),
            "expected corrective shape error, got: {err}"
        );
    }

    struct NormalizedActionCase {
        name: &'static str,
        action: serde_json::Value,
        args: serde_json::Value,
        expected: serde_json::Value,
    }

    fn check_normalized_actions(cases: &[NormalizedActionCase]) {
        for case in cases {
            let (normalized, note) = normalize_action(case.action.clone(), &case.args)
                .unwrap_or_else(|e| {
                    panic!("{}: expected normalization to succeed, got: {e}", case.name)
                });
            assert_eq!(
                normalized, case.expected,
                "{}: normalized action mismatch",
                case.name
            );
            assert!(
                note.is_some(),
                "{}: recoverable shape should report a normalization note",
                case.name
            );
        }
    }

    /// A flat action with siblings, a stringified JSON action and a bare open value all
    /// normalize to the action they name and report a note.
    #[test]
    fn normalize_accepts_json_action_shapes() {
        check_normalized_actions(&[
            // {"action":"open","url":"..."} → {"open":{"url":"..."}}
            NormalizedActionCase {
                name: "flat open with a url sibling",
                action: json!("open"),
                args: json!({"action":"open","url":"https://example.com","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            // {"action":"open","open":{"url":"..."}} → same
            NormalizedActionCase {
                name: "flat open with a nested open",
                action: json!("open"),
                args: json!({"action":"open","open":{"url":"https://example.com"},"tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            // {"action":"open","open":"https://..."} → {"open":{"url":"..."}}
            NormalizedActionCase {
                name: "flat open with a url string under open",
                action: json!("open"),
                args: json!({"action":"open","open":"https://example.com","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            NormalizedActionCase {
                name: "snapshot with no args",
                action: json!("snapshot"),
                args: json!({"action":"snapshot","tab":"t"}),
                expected: json!({"snapshot":{}}),
            },
            NormalizedActionCase {
                name: "get_text with a flattened selector",
                action: json!("get_text"),
                args: json!({"action":"get_text","selector":"body","tab":"t"}),
                expected: json!({"get_text":{"selector":"body"}}),
            },
            // {"action":"{\"open\":{\"url\":\"...\"}}"} → {"open":{"url":"..."}}
            NormalizedActionCase {
                name: "stringified json action",
                action: json!("{\"open\": {\"url\": \"https://example.com\"}}"),
                args: json!({"action":"{\"open\": {\"url\": \"https://example.com\"}}","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            NormalizedActionCase {
                name: "cdata-wrapped json action",
                action: json!("<![CDATA[{\"open\": {\"url\": \"https://example.com\"}}]]>"),
                args: json!({"action":"<![CDATA[{\"open\": {\"url\": \"https://example.com\"}}]]>","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            // {"action":{"open":"https://..."}} → {"open":{"url":"..."}}
            NormalizedActionCase {
                name: "bare open value",
                action: json!({"open":"https://example.com"}),
                args: json!({"action":{"open":"https://example.com"},"tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            NormalizedActionCase {
                name: "bare url string as the whole action",
                action: json!("https://example.com"),
                args: json!({"action":"https://example.com","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
        ]);
    }

    /// XML-wrapped actions normalize too: the tag text decodes to the serde field types.
    #[test]
    fn normalize_accepts_xml_action_shapes() {
        check_normalized_actions(&[
            NormalizedActionCase {
                name: "xml open",
                action: json!("<open><url>https://example.com</url></open>"),
                args: json!({"action":"<open><url>https://example.com</url></open>","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            NormalizedActionCase {
                name: "xml open with a text body",
                action: json!("<open>https://example.com</open>"),
                args: json!({"action":"<open>https://example.com</open>","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            // whitespace inside tags
            NormalizedActionCase {
                name: "xml open with whitespace inside tags",
                action: json!("<open>\n<url>https://example.com</url>\n</open>"),
                args: json!({"action":"<open>\n<url>https://example.com</url>\n</open>","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            // stray closing tag
            NormalizedActionCase {
                name: "xml open with a stray closing tag",
                action: json!("<open><url>https://example.com</url></open></action>"),
                args: json!({"action":"<open><url>https://example.com</url></open></action>","tab":"t"}),
                expected: json!({"open":{"url":"https://example.com"}}),
            },
            NormalizedActionCase {
                name: "xml find",
                action: json!("<find><by>first</by><value>a</value><action>click</action></find>"),
                args: json!({"action":"<find><by>first</by><value>a</value><action>click</action></find>","tab":"t"}),
                expected: json!({"find":{"by":"first","value":"a","action":"click"}}),
            },
            // find with exact/index — XML text decoded to the serde bool/u32 types
            NormalizedActionCase {
                name: "xml find with exact",
                action: json!(
                    "<find><by>text</by><value>submit</value><action>click</action><exact>true</exact></find>"
                ),
                args: json!({"action":"<find><by>text</by><value>submit</value><action>click</action><exact>true</exact></find>","tab":"t"}),
                expected: json!({"find":{"by":"text","value":"submit","action":"click","exact":true}}),
            },
            NormalizedActionCase {
                name: "xml find with index",
                action: json!(
                    "<find><by>nth</by><value>input</value><action>click</action><index>2</index></find>"
                ),
                args: json!({"action":"<find><by>nth</by><value>input</value><action>click</action><index>2</index></find>","tab":"t"}),
                expected: json!({"find":{"by":"nth","value":"input","action":"click","index":2}}),
            },
        ]);
    }

    #[test]
    fn missing_or_empty_tab_defaults_to_logical_default() {
        // Missing tab → the stable logical default, reported as a note.
        // (Per-run physical uniqueness of that default is covered by
        // session_resolution_namespaces_per_agent_run.)
        let (tab, note) =
            ChromeTool::normalize_tab(&json!({"action":{"open":{"url":"https://example.com"}}}));
        assert_eq!(tab, DEFAULT_TAB);
        assert!(note.is_some(), "defaulting should be reported as a note");
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
        // The shared non-agent instance sanitizes the name the same way — link
        // enrichment's own plain names pass through unchanged, keeping their
        // daemon-sweepable namespace.
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
        // The namespace a physical name carries back is the one it was built
        // with: the scheme is read in both directions, never re-derived.
        assert_eq!(session_namespace(&physical), run_session_namespace("run-a"));
        assert_eq!(session_namespace("link-enricher-3"), "link-enricher-3");
    }

    #[test]
    fn agent_sessions_are_never_orphan_swept() {
        // The namespace keeps the protected `agent-tab-` root, so the
        // daemon's orphan-protection sweep can never close a live run's
        // session; the run's own release (through the browser's extension)
        // reclaims them, not this sweep.
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
        // and daemon-wrapper text) fails fast with the unreachable-tab guidance
        // but does NOT mark the daemon unhealthy — the relay and daemon are up,
        // so recovery must not wake for it.
        let err = ChromeTool::fail_fast_if_daemon_down(
            "Auto-launch failed: Could not drive your Chrome through the ab-connect extension. \
             The tab this session was driving can no longer be resolved (it was closed, or a \
             flaky relay dropped it)",
            Some("browser_not_launched"),
            "agent-tab-x-default",
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("the product closes a session's tabs itself"),
            "expected the tabs left to the product, got: {err}"
        );
        assert!(crate::tools::chrome_daemon::is_advertised());
        // Daemon-unavailable signature → actionable guidance, daemon marked
        // unhealthy (wakes the auto-recovery watchdog).
        let err = ChromeTool::fail_fast_if_daemon_down(
            "Failed to read: Resource temporarily unavailable (os error 35) (after 5 retries - daemon may be busy or unresponsive)",
            None,
            "agent-tab-x-default",
        )
        .await
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
            "agent-tab-x-default",
        )
        .await
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
                "agent-tab-x-default",
            )
            .await
            .is_ok()
        );
        assert!(
            ChromeTool::fail_fast_if_daemon_down(
                "chrome-use error: Element not found",
                None,
                "agent-tab-x-default",
            )
            .await
            .is_ok()
        );
        assert!(
            ChromeTool::fail_fast_if_daemon_down(
                "chrome-use error: timed out",
                Some("timeout"),
                "agent-tab-x-default",
            )
            .await
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
                "upload" => json!({"selector": "#f", "files": ["/tmp/a.txt"]}),
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
            "upload",
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
    fn build_args_rejects_verbs_handled_in_execute() {
        for action in [
            ChromeAction::Screenshot {},
            ChromeAction::Upload {
                selector: "#f".to_string(),
                files: vec!["/tmp/a.txt".to_string()],
            },
        ] {
            assert!(
                ChromeTool::build_args(&action).is_err(),
                "{action:?} must be handled in execute(), not build_args"
            );
        }
    }

    #[expect(clippy::too_many_lines)] // the per-argv wait/expect shapes are one cohesive table
    #[test]
    fn build_args_for_wait_and_expect() {
        fn argv(args: &[String]) -> Vec<&str> {
            args.iter().map(String::as_str).collect()
        }

        // Wait: exactly one target; the forwarded --timeout is the declared
        // chrome-side clock, and the product's kill rides the relay-recovery
        // window + slack above it ([`ChromeTool::call_clocks`]).
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
        let resp = match parse_run_output(Some("expect"), false, false, &body.to_string(), "") {
            ParsedRun::Ok(resp) => resp,
            other => panic!("expected Ok, got {other:?}"),
        };
        let outcome = expect_outcome(resp.data.as_ref().expect("data present")).expect("verdict");
        assert!(!outcome.pass && outcome.timed_out);
        assert_eq!(outcome.actual, Some(json!(3)));

        // …but only for expect: the same payload on another action is a failure.
        assert!(matches!(
            parse_run_output(Some("open"), false, false, &body.to_string(), ""),
            ParsedRun::Failed { .. }
        ));

        // A failure envelope on any exit → Failed with its details.
        let err_body = json!({"success": false, "error": "Browser not launched", "code": "browser_not_launched"});
        match parse_run_output(Some("expect"), false, false, &err_body.to_string(), "") {
            ParsedRun::Failed { error, code, .. } => {
                assert_eq!(error, "Browser not launched");
                assert_eq!(code.as_deref(), Some("browser_not_launched"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }

        // Unparseable stdout: zero exit → Unparseable; non-zero → stderr fallback.
        assert!(matches!(
            parse_run_output(None, false, true, "garbage", ""),
            ParsedRun::Unparseable
        ));
        match parse_run_output(None, false, false, "garbage", "relay is down\n") {
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

        // A process the command left behind writing into the inherited channel
        // after chrome-use's own answer: the trailing bytes are ignored, so a
        // zero-exit call stays the success its envelope says it is.
        let with_leftover = format!(
            "{}\nsomeone the command left behind is still writing\n",
            json!({"ok": true, "data": {"text": "hi"}})
        );
        assert!(matches!(
            parse_run_output(None, false, true, &with_leftover, ""),
            ParsedRun::Ok(_)
        ));

        // An answer the product cut off at its cap names itself whatever the
        // exit status: the truncation, not chrome-use's exit, is the reason.
        assert!(matches!(
            parse_run_output(None, true, false, "garbage", "relay is down\n"),
            ParsedRun::Unparseable
        ));

        // Happy path.
        assert!(matches!(
            parse_run_output(
                Some("open"),
                false,
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
        );
        assert!(out.contains("hello"), "output: {out}");
        assert!(
            !out.contains('{'),
            "should not pretty-print JSON braces: {out}"
        );
    }

    /// `readBack` without a `warning` is chrome-use's normal success shape —
    /// no note may be appended. A warning on a type read-back mismatch (exit 0)
    /// must be surfaced, unquoted, and marked not applied.
    #[test]
    fn format_action_output_warning_cases() {
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
        );
        assert!(
            !out.contains("chrome-use warning"),
            "clean output must not carry a warning: {out}"
        );

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

        // A non-screenshot action never re-attaches the last screenshot: a stale
        // path on the tool must not become this action's image payload.
        *tool.last_screenshot.lock().unwrap_poison() = Some("/tmp/stale.png".to_string());
        let non_screenshot = tool
            .image_payload(
                &crate::Workspace::default(),
                &serde_json::json!({"action": "open", "url": "https://example.com", "tab": "default"}),
            )
            .await;
        assert!(
            non_screenshot.is_none(),
            "non-screenshot action must yield no payload"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `upload` answer's four rules: chrome-use's own counts and the absolute
    /// file each path was read from; a stated zero never reads as delivery; a
    /// consumed input stays a success with its warning; and a skipped one-file
    /// check is a note on that same success.
    #[test]
    fn upload_output_reports_delivery_and_keeps_the_warning_a_success() {
        let resolved = vec![PathBuf::from("/tmp/a.txt"), PathBuf::from("/tmp/b.txt")];
        let kept = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "uploaded": 2, "attached": 2 })),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::upload_output("default", "#f", &resolved, &kept, false);
        assert!(out.contains("/tmp/a.txt, /tmp/b.txt"), "{out}");
        assert!(out.contains("still holds them"), "{out}");
        assert!(!out.contains("⚠"), "{out}");

        // A file input the page emptied after the upload warns on an exit-0
        // success: the files did reach the page, so the answer repeats the browser's
        // words and never calls the attach failed.
        let consumed = ChromeResponse {
            success: Some(true),
            data: Some(json!({
                "uploaded": 2, "attached": 0,
                "warning": "the file input is now empty",
            })),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::upload_output("default", "#f", &resolved, &consumed, false);
        assert!(out.contains("the file input is now empty"), "{out}");
        assert!(out.contains("not a failed attach"), "{out}");
        assert!(!out.contains("still holds them"), "{out}");

        // A stated zero never reads as delivery, and never claims the input kept
        // files the browser says were not uploaded.
        let nothing = ChromeResponse {
            success: Some(true),
            data: Some(json!({
                "uploaded": 0, "attached": 0,
                "warning": "the files could not be streamed",
            })),
            ..ChromeResponse::default()
        };
        let out = ChromeTool::upload_output("default", "#f", &resolved, &nothing, false);
        assert!(out.contains("Nothing reached the page"), "{out}");
        assert!(out.contains("the files could not be streamed"), "{out}");
        assert!(!out.contains("Delivered"), "{out}");
        assert!(!out.contains("still holds them"), "{out}");
        assert!(!out.contains("did reach the page"), "{out}");

        // The one-file check the browser could not run is a note, not a failure:
        // the delivery it could not judge is still the answer.
        let out = ChromeTool::upload_output("default", "#f", &resolved, &kept, true);
        assert!(out.contains("Note:"), "{out}");
        assert!(
            out.contains(crate::chrome::files::unchecked_several_files_note()),
            "{out}"
        );
    }

    // ── the per-agent tab cap: what counts as a tab, and what a closed tab says ─────

    /// The one failure that must not count as a tab: a call that left no page in the browser at
    /// all. Everything the page itself answered — a missing element, a failed navigation, a
    /// chrome-use timeout on a page that is there — is a page the session has.
    #[test]
    fn a_call_that_left_no_page_creates_no_tab() {
        for message in [
            "relay isn't connected to the extension",
            "daemon may be busy or unresponsive",
            "chrome-use CLI is not available. install it",
            "Failed to execute chrome-use CLI",
            "chrome-use did not answer within 45s for `open`",
            self_launched_browser_error(
                "This session's previous browser is gone; a fresh one was launched instead.",
            )
            .as_str(),
            // The browser answered that the session owns no page under this name — a tab it
            // cannot drive is no tab of the record's, whichever way it was lost.
            "session owns no resolvable tab. The chrome-use extension lost its debugger attach",
        ] {
            assert!(
                ChromeTool::left_no_page(&anyhow::anyhow!("{message}")),
                "no page can exist for: {message}"
            );
        }
        for message in [
            "element not found: #missing",
            "Navigation to https://example.com failed — Chrome rendered its error page: \
             ERR_NAME_NOT_RESOLVED (name not resolved). Verify the URL and network.",
            "Wait timed out after 10000ms",
            "Navigation failed: the tab is still on a blank page after opening https://x",
        ] {
            assert!(
                !ChromeTool::left_no_page(&anyhow::anyhow!("{message}")),
                "the session really has this page: {message}"
            );
        }
    }

    /// The group tail runs after a failed step unless the failure proves the owner's browser
    /// was never reached: a step cut off at a clock may have opened its tab first, so its group
    /// must still be settled.
    #[test]
    fn a_cut_off_step_still_owes_its_group_a_settle() {
        for message in [
            "chrome-use did not answer within 45s for `open`",
            "command timed out after 30000ms",
            "event stream closed before the call was answered",
            "session owns no resolvable tab. The chrome-use extension lost its debugger attach",
            "Navigation to https://example.com failed — Chrome rendered its error page: \
             ERR_NAME_NOT_RESOLVED (name not resolved). Verify the URL and network.",
        ] {
            assert!(
                !ChromeTool::never_reached_the_owner_browser(message),
                "the step may have opened its tab before this failure: {message}"
            );
        }
    }

    /// A tab the cap closed is refused from the tool's own record, before the browser is asked
    /// anything: the agent reads that the tab is gone and how to get it back — never an
    /// unreachable browser, never a lost extension attach.
    #[tokio::test]
    #[serial_test::serial(chrome_tabs)]
    async fn a_tab_the_cap_closed_is_refused_before_the_browser_is_asked() {
        let _ledger = crate::tools::chrome_tab_ledger::TestLedgerGuard::install();
        let tool = ChromeTool::new(ChromeRunSessions::for_run("run-cap-refusal"));
        let namespace = tool.session_prefix.clone();
        // Fill the cap, then let one more tab appear: the tab the run has addressed least
        // recently ("docs", addressed first) is the one the record closes.
        for tab in ["docs", "a", "b", "c", "d"] {
            assert!(
                crate::tools::chrome_tab_ledger::note_addressed(&namespace, tab, true).is_none()
            );
        }
        let eviction = crate::tools::chrome_tab_ledger::note_addressed(&namespace, "e", true)
            .expect("the cap was crossed");
        assert_eq!(eviction.name, "docs");
        crate::tools::chrome_tab_ledger::settle_eviction(&namespace, &eviction, true);
        assert!(crate::tools::chrome_tab_ledger::is_closed(
            &namespace, "docs"
        ));

        let error = tool
            .execute(
                &crate::Workspace::default(),
                json!({"action": "snapshot", "tab": "docs"}),
            )
            .await
            .expect_err("a tab the cap closed is refused");
        let message = error.to_string();
        assert!(
            message.contains("docs") && message.contains("no longer exists"),
            "the refusal names the tab and says it is gone: {message}"
        );
        assert!(
            !message.contains("extension") && !message.contains("debugger"),
            "the refusal is the tool's own record, never a lost extension attach: {message}"
        );
        // Only the tab the cap closed is refused; the run's other tabs are ordinary.
        assert!(!tool.tab_was_closed_by_cap("a"));
        // The record is keyed by the logical name, so the same tab spelled as its physical
        // session name is the same refusal rather than a second tab.
        assert!(tool.tab_was_closed_by_cap(&format!("{namespace}docs")));
    }
}
