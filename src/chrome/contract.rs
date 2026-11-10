//! One place for the chrome-use response envelope parsing, shared by the interactive
//! `chrome` tool and the `mahbot chrome` CLI.
//!
//! chrome-use answers `--json` commands with either the classic `success` key or the newer `ok`
//! key, so [`ChromeResponse`] is the single structural parser every frontend shares, and
//! [`ChromeResponse::verdict`] applies the shared failure precedence so the two paths cannot
//! drift. The `mahbot chrome` stdout contract ([`SCHEMA_VERSION`], [`OutKind`],
//! [`OutEnvelope`]) lives here too.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::LazyLock;

/// Response from chrome-use `--json` commands — the ONE structural envelope parser every
/// chrome frontend shares. There is no schema-version marker in chrome-use and the envelope
/// even varies within one binary (`session list` answers `ok:true`, everything else
/// `success:true`), so tolerance is structural: every field is [`Option`].
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChromeResponse {
    /// Classic `success` envelope key — absent on newer `ok`-keyed envelopes.
    #[serde(default)]
    pub(crate) success: Option<bool>,
    /// Latest chrome-use envelopes replace `success` with `ok` on some commands.
    #[serde(default)]
    pub(crate) ok: Option<bool>,
    pub(crate) data: Option<Value>,
    pub(crate) error: Option<String>,
    /// Stable error-envelope code (v1.5.78+) — present on structured errors.
    pub(crate) code: Option<String>,
    /// `retryable` flag on structured error envelopes, surfaced to the agent as a retry hint.
    #[serde(default)]
    pub(crate) retryable: Option<bool>,
    /// Any other top-level keys, so chrome-use's degraded-success `warning` field is visible
    /// wherever it puts it ([`chrome_use_warning`]).
    #[serde(flatten)]
    pub(crate) extra: serde_json::Map<String, Value>,
}

impl ChromeResponse {
    /// Success gate accepting both the classic `success: true` and the `ok: true` envelope.
    pub(crate) fn is_success(&self) -> bool {
        envelope_success(self.success, self.ok)
    }

    /// Tri-state verdict: `Some(true)` when either key reports success, `Some(false)` when
    /// either reports failure, `None` when the payload carries no verdict key (callers own their
    /// default: strict / tolerance-first / error-first).
    pub(crate) fn verdict(&self) -> Option<bool> {
        if self.success.is_none() && self.ok.is_none() {
            return None;
        }
        Some(self.is_success())
    }

    /// The single `Value`-path parse entry: unknown keys are captured in `extra` and any
    /// non-object or unparseable payload yields a default response.
    #[must_use]
    pub(crate) fn from_value(v: &Value) -> Self {
        serde_json::from_value(v.clone()).unwrap_or_default()
    }
}

/// chrome-use's `warning` field — the degraded-success signal for `type` (the page rewrote or
/// filtered the text) and `press` (the key reached nothing). It rides inside `data` or at the
/// envelope top level (the flattened `extra` map), and chrome-use emits it ONLY on a degraded
/// exit-0 success, so its presence means "the action may not have taken effect".
pub(crate) fn chrome_use_warning(resp: &ChromeResponse) -> Option<Value> {
    let sources: [Option<&serde_json::Map<String, Value>>; 2] = [
        Some(&resp.extra),
        resp.data.as_ref().and_then(Value::as_object),
    ];
    sources
        .into_iter()
        .flatten()
        .find_map(|map| map.get("warning").cloned())
}

/// A warning's text as it is shown to a caller: a plain string unquoted, any other value as its
/// JSON.
pub(crate) fn warning_text(warning: &Value) -> String {
    warning
        .as_str()
        .map_or_else(|| warning.to_string(), str::to_string)
}

/// Stable fragments of chrome-use's browser-replacement note, attached to a command it ran in a
/// browser it launched ITSELF. Matched as fragments rather than one whole sentence: the
/// parenthesized reason is chrome-use's own diagnosis and drifts with it.
const SELF_LAUNCHED_BROWSER_PHRASES: [&str; 2] =
    ["previous browser is gone", "a fresh one was launched"];

/// The browser-replacement note on a chrome-use envelope: `Some(text)` when this command ran in
/// a browser chrome-use launched itself instead of the owner's real one.
///
/// Deliberately distinct from the degraded-success warnings the same field carries: those mean
/// the action may not have taken effect in the browser the model was driving, this one means the
/// browser was not the owner's at all, so callers must report it as a plain failure
/// ([`self_launched_browser_error`]).
#[must_use]
pub(crate) fn self_launched_browser_note(resp: &ChromeResponse) -> Option<String> {
    let warning = chrome_use_warning(resp)?;
    let text = warning
        .as_str()
        .map_or_else(|| warning.to_string(), str::to_string);
    let lower = text.to_ascii_lowercase();
    SELF_LAUNCHED_BROWSER_PHRASES
        .iter()
        .any(|phrase| lower.contains(phrase))
        .then_some(text)
}

/// Opening sentence of [`self_launched_browser_error`], so a caller that must recognize that
/// failure never re-spells it.
const SELF_LAUNCHED_BROWSER_PREFIX: &str =
    "chrome-use launched a browser of its own for this command";

/// The failure a call whose envelope carried [`self_launched_browser_note`] gets: the tool's own
/// note, what it means, and — per audience — what to do next.
///
/// This is NOT the daemon-down path: the daemon and relay may be perfectly healthy, and what was
/// lost is the connection to the owner's real browser, so the call must never be reported as a
/// quiet success and the daemon must not be marked unhealthy for it.
#[must_use]
fn self_launched_browser_diagnosis(note: &str) -> String {
    format!(
        "{SELF_LAUNCHED_BROWSER_PREFIX} ({note}). The work ended up in \
         a browser chrome-use started itself, not the owner's real logged-in Chrome — the tool \
         could not resolve the connection to the real browser, so the page state this action \
         assumed is not there and nothing read or written here happened in the owner's session."
    )
}

/// The tool's copy of the diagnosis, for the agent: retry, then hand the diagnosis over as it
/// came — repairing the extension or the relay is the owner's side.
#[must_use]
pub(crate) fn self_launched_browser_error(note: &str) -> String {
    format!(
        "{}. Retry the action; if it keeps happening, report it as it came — the extension and \
         the relay are the owner's side to repair, not this tool's.",
        self_launched_browser_diagnosis(note)
    )
}

/// The CLI's copy of the diagnosis, for the person reading the failure: it keeps the repair path
/// the tool's copy deliberately leaves out.
#[must_use]
pub(crate) fn self_launched_browser_cli_error(note: &str) -> String {
    format!(
        "{}. Retry the action; if it keeps happening, check the extension at \
         chrome://extensions and the relay, then retry.",
        self_launched_browser_diagnosis(note)
    )
}

/// Whether `message` is the failure [`self_launched_browser_error`] built.
#[must_use]
pub(crate) fn is_self_launched_browser_error(message: &str) -> bool {
    message.starts_with(SELF_LAUNCHED_BROWSER_PREFIX)
}

/// The failure text for output the product had to cut off, either at the bytes it collects
/// ([`crate::chrome::spawn::PIPE_READ_CAP`]) or when the collection window ended mid-stream:
/// that is a different fact from chrome-use answering something malformed.
#[must_use]
pub(crate) fn truncated_output_error() -> String {
    format!(
        "the answer chrome-use wrote was cut off by mahbot before it could be read in full — it \
         exceeded the {} MiB mahbot collects per stream, or a process the command left behind kept \
         writing past the collection window — so it could not be parsed. The command itself may \
         well have succeeded: retry with a narrower action (a smaller selector or page region) \
         rather than reading this as malformed output from chrome-use.",
        crate::chrome::spawn::PIPE_READ_CAP / (1024 * 1024)
    )
}

/// Parse the FIRST JSON value chrome-use wrote, ignoring anything after it — the ONE reading of
/// a chrome-use answer.
///
/// A process the command left behind can write into the inherited output channel after
/// chrome-use's own answer, and those trailing bytes must not turn a finished command's answer
/// into malformed output.
pub(crate) fn parse_first<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    serde_json::Deserializer::from_slice(bytes)
        .into_iter::<T>()
        .next()?
        .ok()
}

/// The note a caller appends to a chrome-use result whose child exited while a process it left
/// behind still held the call's output channel
/// ([`crate::chrome::spawn::CliOutput::leftover_pipes`]) — on a FAILING result too, where the
/// failure stays the failure.
///
/// What still runs is chrome-use's OWN helper, the daemon it starts for the session, which
/// chrome-use stops on its own idle timeout — so a caller that leaves it alone loses nothing.
#[must_use]
pub(crate) fn chrome_leftover_pipe_note(session: &str) -> String {
    format!(
        "[leftover helper] a helper chrome-use starts itself — the daemon of chrome-use session \
         `{session}` — still holds the call's output channel, so no more of its output could be \
         collected. chrome-use starts that helper on its own and stops it on its own idle timeout; \
         to stop it now, run `{}`.",
        chrome_session_stop_command(session)
    )
}

/// The `session stop` command that addresses chrome-use session `session`'s leftover helper —
/// the recipe [`chrome_leftover_pipe_note`] embeds and the durable record carries as `detail`.
///
/// The name is rendered bare, and that is safe: both producers and chrome-use's own
/// `validation::is_valid_session_name` restrict it to `[A-Za-z0-9_-]`, characters that are inert
/// in a shell, and the names are prefix-anchored — so the recipe cannot be turned into a
/// different command by a name. Quoting is deliberately not used: the recipe is run by the
/// agent's shell, which may be `cmd.exe`, where the unix single-quote form would become part of
/// the argument. `--force` is part of it because the interactive tool's sessions live under the
/// protected `agent-tab-*` prefix.
#[must_use]
pub(crate) fn chrome_session_stop_command(session: &str) -> String {
    format!("mahbot chrome session stop {session} --force")
}

/// Core envelope-success predicate with failure precedence: an explicit `false` on either key
/// loses over a contradicting success key, and a payload with neither key is not a success.
fn envelope_success(success: Option<bool>, ok: Option<bool>) -> bool {
    !(success == Some(false) || ok == Some(false)) && (success == Some(true) || ok == Some(true))
}

/// Extract textual content from a chrome-use snapshot response `data` field: a plain string
/// (`snapshot -c`), an object with `text` (`get_text`), `result` (`eval`) or `content`
/// (`get_text`), or the `origin`/`refs`/`snapshot` shape of `snapshot --json`. `None` when none
/// of these match (the caller falls back).
pub(crate) fn extract_snapshot_text(data: &serde_json::Value) -> Option<String> {
    data.as_str().map(String::from).or_else(|| {
        ["text", "result", "content", "snapshot"]
            .iter()
            .find_map(|key| data.get(*key).and_then(Value::as_str).map(String::from))
    })
}

/// Unwrap the actual JS result from an eval response: `data = {origin, result}` with BOTH keys
/// is that wrapper, every other shape passes through.
pub(crate) fn eval_result(resp: &ChromeResponse) -> Option<&Value> {
    resp.data
        .as_ref()
        .map(|d| match (d.get("result"), d.get("origin")) {
            (Some(result), Some(_)) => result,
            _ => d,
        })
}

/// The ONE error-page probe eval, shared by the CLI `open` action and the interactive tool:
/// always a JSON verdict string, reporting whether the tab committed to a Chrome error page and,
/// best-effort, the net error token from `div.error-code` — empty until the neterror script
/// renders it, so callers must keep the generic fallback.
pub(crate) const ERROR_PAGE_PROBE_JS: &str = "JSON.stringify({err: location.protocol === 'chrome-error:', code: (document.querySelector('.error-code') || {}).textContent || ''})";

/// Parsed [`ERROR_PAGE_PROBE_JS`] verdict.
#[derive(Debug, PartialEq)]
pub(crate) struct ErrorPageProbe {
    /// The tab conclusively sits on a Chrome error page.
    pub(crate) is_error_page: bool,
    /// The net error token (`ERR_*` / `DNS_*`) read from `div.error-code`, when recognizable.
    pub(crate) code: Option<String>,
}

/// Parse an error-page probe response. `None` = inconclusive (callers treat the probe as passed,
/// never as a network failure).
pub(crate) fn parse_error_page_probe(resp: &ChromeResponse) -> Option<ErrorPageProbe> {
    let data = eval_result(resp)?;
    let verdict = if let Some(text) = extract_snapshot_text(data) {
        let t = text.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("null") || t.eq_ignore_ascii_case("undefined") {
            return None;
        }
        serde_json::from_str::<Value>(t)
            .ok()
            .or_else(|| extract_net_error_code(t).map(|c| json!({ "err": true, "code": c })))
    } else if let v @ Value::Object(_) = data {
        Some(v.clone())
    } else {
        None
    }?;
    let is_error_page = verdict.get("err").and_then(Value::as_bool) == Some(true);
    let code = is_error_page.then(|| {
        verdict
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or_default()
    });
    Some(ErrorPageProbe {
        is_error_page,
        code: code.and_then(extract_net_error_code),
    })
}

pub(crate) fn eval_count(resp: &ChromeResponse) -> Option<u64> {
    let data = eval_result(resp)?;
    data.as_u64()
        .or_else(|| data.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
}

/// Trim `rows` to `limit` while reporting the honest `total`. Returns `(trimmed, total, returned)`.
fn trim_extract_rows(rows: Vec<Value>, limit: Option<usize>) -> (Vec<Value>, usize, usize) {
    let total = rows.len();
    let returned = limit.map_or(total, |l| l.min(total));
    let trimmed: Vec<Value> = rows.into_iter().take(returned).collect();
    (trimmed, total, returned)
}

/// Rows from a chrome-use `extract --json` response `data` field:
/// `data = {extracted: [...], count, origin, meta}`. Tolerance: a plain array passes through, a
/// bare object is a single-object extract and becomes a one-element list.
fn extract_rows(data: &Value) -> Vec<Value> {
    match data {
        Value::Object(o) => match o.get("extracted") {
            Some(Value::Array(a)) => a.clone(),
            Some(other) => vec![other.clone()],
            None => vec![Value::Object(o.clone())],
        },
        Value::Array(a) => a.clone(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    }
}

/// The ONE extract output contract for both frontends: `data` carries the (possibly trimmed)
/// rows, `total` the honest full count, and `returned` appears only when a `limit` was applied.
#[must_use]
pub(crate) fn extract_output(data: &Value, limit: Option<usize>) -> Value {
    let (trimmed, total, returned) = trim_extract_rows(extract_rows(data), limit);
    let mut payload = json!({ "data": trimmed, "total": total });
    if limit.is_some() {
        payload["returned"] = json!(returned);
    }
    payload
}

/// A parsed `expect` verdict. `pass` is the authoritative tri-state anchor: `None` from
/// [`expect_outcome`] means the payload was not an expect verdict at all.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpectOutcome {
    pub(crate) pass: bool,
    /// The observed value chrome-use evaluated — `None` when absent.
    pub(crate) actual: Option<Value>,
    /// chrome-use sets `timedOut: true` when the condition never held within the deadline. mahbot
    /// never passes --no-wait, so a false verdict arrives folded into `timedOut`; a bare
    /// `pass:false` is defensive against a future grammar change.
    pub(crate) timed_out: bool,
}

/// Parse the `expect` verdict out of a response `data` value. `None` when the payload carries no
/// boolean `pass` key.
#[must_use]
pub(crate) fn expect_outcome(data: &Value) -> Option<ExpectOutcome> {
    let obj = data.as_object()?;
    let pass = obj.get("pass")?.as_bool()?;
    Some(ExpectOutcome {
        pass,
        actual: obj.get("actual").cloned(),
        timed_out: obj
            .get("timedOut")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

/// `mahbot chrome` stdout envelope schema version.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Failure/success classification of one CLI step — drives the emitted `kind` and the exit code
/// (site problems are rc 1 and schedulable; environment problems are rc 2 and need a human).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutKind {
    Ok,
    Empty,
    Timeout,
    Network,
    Redesign,
    NotFound,
    Error,
    Environment,
    Usage,
}

impl OutKind {
    /// Stable `kind` string emitted on the envelope.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Empty => "empty",
            Self::Timeout => "timeout",
            Self::Network => "network",
            Self::Redesign => "redesign",
            Self::NotFound => "not-found",
            Self::Error => "error",
            Self::Environment => "environment",
            Self::Usage => "usage",
        }
    }

    /// Process exit code for this kind: site/data problems rc 1 and retryable, environment
    /// problems rc 2, usage rc 3.
    #[must_use]
    pub(crate) const fn exit_code(self) -> i32 {
        match self {
            Self::Ok | Self::Empty => 0,
            Self::Timeout | Self::Network | Self::Redesign | Self::NotFound | Self::Error => 1,
            Self::Environment => 2,
            Self::Usage => 3,
        }
    }
}

// Failure signatures live here (not in chrome_daemon) so the interpretation layer stays
// dependency-free: chrome_daemon imports them back, and both frontends classify through this
// module without any frontend-to-frontend coupling.

/// Per-session wedge signature: the session's CDP attach is unresponsive and every command on
/// that named session times out while the daemon and relay may be healthy. These texts are
/// DaemonWedge signals and MUST keep triggering the watchdog's auto-recovery; the CLI uses this
/// matcher directly to attach the wedge-specific hint.
pub(crate) fn is_session_unresponsive_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("session unresponsive")
        || lower.contains("cdp session is unresponsive after attaching")
}

/// Detect the daemon-unavailable signature chrome-use produces when its background daemon is dead
/// or wedged: the CLI hangs in its own 5-retry loop and eventually reports "daemon may be busy or
/// unresponsive". Also covers the 1.5.8x-era texts and — via [`is_session_unresponsive_error`] —
/// the session-wedge texts.
pub(crate) fn is_daemon_unavailable_error(msg: &str) -> bool {
    // The session-wedge texts are also DaemonWedge signals for the watchdog's auto-recovery.
    if is_session_unresponsive_error(msg) {
        return true;
    }
    let lower = msg.to_ascii_lowercase();
    lower.contains("resource temporarily unavailable")
        || lower.contains("os error 35")
        || lower.contains("os error 11")
        || lower.contains("daemon may be busy or unresponsive")
        || lower.contains("daemon failed to start")
        || lower.contains("auto-launch failed")
        // Colon form only — "failed to connect to <host>" is a page-level navigation failure.
        || lower.contains("failed to connect:")
}

/// Relay-side failure signature — the daemon is alive but cannot drive Chrome through the
/// extension relay. A wedge clears on restart; a relay drop needs the extension to republish.
pub(crate) fn is_relay_unavailable_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("relay isn't connected")
        || lower.contains("relay is not")
        || lower.contains("relay dropped")
        || lower.contains("relay down")
        || lower.contains("could not drive your chrome")
}

/// Stable error-envelope `code` values that are unambiguously daemon-side. The coarse
/// `connection_failed` code is shared with page-level navigation failures, so the message-text
/// matcher stays the source of truth for those.
pub(crate) fn is_daemon_unavailable_code(code: Option<&str>) -> bool {
    matches!(code, Some("browser_not_launched"))
}

/// Structural failures of a call that never left the product for the owner's browser: the relay
/// signature, the daemon signature (which folds in the session-wedge texts), "browser not
/// launched", "requires ab-connect" and "only works over the extension relay". Everything the
/// browser may have run before the failure is NOT here — [`unreached_browser_error`] is that
/// wider class.
pub(crate) fn never_dispatched_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    is_relay_unavailable_error(msg)
        || is_daemon_unavailable_error(msg)
        || lower.contains("browser not launched")
        || lower.contains("requires ab-connect")
        || lower.contains("only works over the extension relay")
}

/// Structural failures that provably never reached the browser — [`never_dispatched_error`] plus
/// "event stream closed", "command timed out", "stale target" and "target closed".
pub(crate) fn unreached_browser_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    never_dispatched_error(msg)
        || lower.contains("event stream closed")
        || lower.contains("command timed out")
        || lower.contains("stale target")
        || lower.contains("target closed")
}

/// Error signatures of a leftover tab the daemon can no longer re-drive: its binding went stale
/// while the extension keeps the attach, and the extension never re-attaches `about:` URLs. Its
/// session has to move to another name, while the tab itself is the session's and the product
/// closes it. An orphan the extension fully dropped never produces these. The "relay lost it"
/// variant is a permanent orphan; "navigated across processes" is a recoverable OAuth/SSO
/// retarget and must stay OUT.
pub(crate) fn is_unreachable_tab_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("can no longer be resolved")
        || lower.contains("owns no resolvable tab")
        || lower.contains("no attached tab")
        || lower.contains("its tab is gone")
        || lower.contains("stale session")
        || lower.contains("unknown session")
        || lower.contains("the relay lost it")
}

/// Actionable error for a real call that hit an orphaned tab: the daemon and relay are up, only
/// the session's tab is unreachable. Fail fast without marking the daemon unhealthy — recovery
/// cannot fix a Chrome-side orphan — and without asking anyone to close it.
pub(crate) fn unreachable_tab_message(error: &str) -> String {
    format!(
        "{error}. The chrome-use extension lost its debugger attach to this tab and never \
         re-attaches about:blank tabs, so this session cannot drive it again. The tab is \
         this session's, and the product closes a session's tabs itself — at run end, and \
         when a wedged session is recovered — so leave it alone; use another `tab` name \
         for a page this session can drive (the chrome daemon itself is healthy)."
    )
}

/// Classify a chrome-use step failure into an [`OutKind`]. Environment signatures win because
/// rc 2 must not be masked by a page-level wrapper text.
pub(crate) fn classify_call_failure(code: Option<&str>, error: &str) -> OutKind {
    if is_daemon_unavailable_error(error)
        || is_daemon_unavailable_code(code)
        || is_relay_unavailable_error(error)
    {
        return OutKind::Environment;
    }
    if error.contains("net::ERR_") || code == Some("connection_failed") {
        return OutKind::Network;
    }
    // chrome-use's internal action timeout. Chrome-side deadlines equal the declared budget and
    // the mahbot-side kill rides KILL_SLACK above them, so chrome-use's phrased timeouts normally
    // surface first. Matched by its specific phrasings rather than a bare "timed out" substring,
    // so an unrecognized connect-timeout still classifies as Error.
    let lower = error.to_ascii_lowercase();
    if code == Some("timeout")
        || lower.contains("timed out after")
        || lower.contains("waitfor timed out")
    {
        return OutKind::Timeout;
    }
    if code == Some("element_not_found") || lower.contains("element not found") {
        return OutKind::NotFound;
    }
    OutKind::Error
}

/// Canonical net error token extracted from chrome-side text: the `net::ERR_*` prefix form on CDP
/// `errorText` and the bare `ERR_*`/`DNS_*` form Chrome renders both canonicalize to the token.
pub(crate) fn extract_net_error_code(text: &str) -> Option<String> {
    static NET_ERROR_CODE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?:net::)?\b(ERR_[A-Z0-9_]+|DNS_[A-Z0-9_]+)").expect("valid net error regex")
    });
    NET_ERROR_CODE
        .captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// Coarse cause phrase for a net error token: DNS failures, refused connections, everything else
/// shares one "unreachable or failed" bucket. `ERR_NAME_NOT_RESOLVED` counts as DNS: it is the
/// code chrome-use surfaces via CDP errorText.
pub(crate) fn net_error_phrase(code: &str) -> &'static str {
    if code.starts_with("DNS_") || code == "ERR_NAME_NOT_RESOLVED" {
        "DNS resolution failure"
    } else if code == "ERR_CONNECTION_REFUSED" {
        "connection refused"
    } else {
        "site unreachable or connection failed"
    }
}

/// Start of the canned remediation hint chrome-use appends to some timeout failures; everything
/// from it to the end of the message is stripped by [`sanitize_timeout_message`].
const STALE_RELAY_HINT_MARKER: &str = "Hint: the session's browser connection is unresponsive";

/// chrome-use appends a canned hint to some timeout failures ("the session's browser connection
/// is unresponsive … Reconnect with `connect`"), which is wrong in mahbot's surface — there is no
/// `connect` verb and the connection is usually healthy — so both frontends strip it from
/// Timeout-classified messages. Environment-classified diagnostics are never rewritten. The
/// wording is chrome-use's: if a release rephrases it the strip degrades to a no-op.
pub(crate) fn sanitize_timeout_message(kind: OutKind, message: &str) -> String {
    if kind != OutKind::Timeout {
        return message.to_string();
    }
    match message.find(STALE_RELAY_HINT_MARKER) {
        Some(idx) => {
            let head = message[..idx].trim_end();
            head.strip_suffix('.').unwrap_or(head).to_string()
        }
        None => message.to_string(),
    }
}

/// Remediation appended to `wait` condition timeouts in BOTH frontends, after the honest
/// chrome-use text — one source so the CLI and the tool never drift.
pub(crate) const WAIT_TIMEOUT_NOTE: &str = " — the target never appeared within the deadline; consider verifying the \
     selector/text/URL or allowing more time";

/// Same for `expect` (whose cause text already names the deadline).
pub(crate) const EXPECT_TIMEOUT_NOTE: &str =
    " — consider verifying the condition or allowing more time";

/// Append the condition-timeout remediation to `message` when `kind` is Timeout and `message` is
/// chrome-use's own phrased text — an empty message is the mahbot-side deadline kill and carries
/// its own factual text. No-op for every other action.
pub(crate) fn with_condition_timeout_note(action: &str, kind: OutKind, message: &mut String) {
    let note = match (kind, action) {
        (OutKind::Timeout, "wait") => Some(WAIT_TIMEOUT_NOTE),
        (OutKind::Timeout, "expect") => Some(EXPECT_TIMEOUT_NOTE),
        _ => None,
    };
    if let Some(note) = note
        && !message.is_empty()
    {
        message.push_str(note);
    }
}

/// The one-line stdout envelope. `payload` must be a JSON object; it is flattened after `kind`.
pub(crate) struct OutEnvelope {
    pub(crate) action: String,
    pub(crate) ok: bool,
    pub(crate) kind: OutKind,
    pub(crate) payload: serde_json::Value,
}

impl OutEnvelope {
    /// Serialize to the single-line stdout envelope JSON.
    #[must_use]
    pub(crate) fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Wire<'a> {
            schema: u32,
            action: &'a str,
            ok: bool,
            kind: &'a str,
            #[serde(flatten)]
            payload: &'a serde_json::Value,
        }
        let wire = Wire {
            schema: SCHEMA_VERSION,
            action: &self.action,
            ok: self.ok,
            kind: self.kind.as_str(),
            payload: &self.payload,
        };
        serde_json::to_string(&wire).expect("OutEnvelope serialization cannot fail")
    }

    /// Write the envelope to stdout as exactly one line.
    pub(crate) fn emit(&self) {
        println!("{}", self.to_json());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chrome_response_accepts_tolerant_envelopes() {
        let ok: ChromeResponse =
            serde_json::from_str(r#"{"ok":true,"data":{}}"#).expect("tolerant deserialize");
        assert!(ok.is_success());

        let success: ChromeResponse =
            serde_json::from_str(r#"{"success":true}"#).expect("tolerant deserialize");
        assert!(success.is_success());

        let failed: ChromeResponse =
            serde_json::from_str(r#"{"success":false}"#).expect("tolerant deserialize");
        assert!(!failed.is_success());

        // Failure precedence — an explicit false on either key loses over a contradicting
        // success key.
        let mixed: ChromeResponse =
            serde_json::from_str(r#"{"success":false,"ok":true}"#).expect("tolerant deserialize");
        assert!(!mixed.is_success());
    }

    /// The tolerant parse keeps a leftover process from deciding the call, while an answer the
    /// product cut short still has no first value to read.
    #[test]
    fn parse_first_ignores_anything_after_the_first_value() {
        let envelope = br#"{"ok":true,"data":{"count":1}}"#;
        assert!(parse_first::<ChromeResponse>(envelope).is_some_and(|r| r.is_success()));

        let mut trailing = envelope.to_vec();
        trailing.extend_from_slice(b"\nsomeone the command left behind is still writing\n");
        let parsed = parse_first::<ChromeResponse>(&trailing).expect("trailing bytes ignored");
        assert_eq!(parsed.data.expect("data present")["count"], 1);
        assert!(parse_first::<Value>(&trailing).is_some());

        // A cut-off answer and a non-JSON one have no first value, so each keeps
        // its own report instead of reading as chrome-use's malformed output.
        assert!(parse_first::<ChromeResponse>(br#"{"ok":true,"data":{"co"#).is_none());
        assert!(parse_first::<ChromeResponse>(b"chrome-use is not installed\n").is_none());
    }

    #[test]
    fn verdict_covers_both_envelopes() {
        let v = |json: serde_json::Value| ChromeResponse::from_value(&json).verdict();
        assert_eq!(v(serde_json::json!({"success": true})), Some(true));
        assert_eq!(v(serde_json::json!({"ok": true})), Some(true));
        assert_eq!(
            v(serde_json::json!({"success": false, "error": "x"})),
            Some(false)
        );
        assert_eq!(
            v(serde_json::json!({"ok": false, "error": "x"})),
            Some(false)
        );
        // Failure beats an unrelated success key; no verdict key → None.
        assert_eq!(
            v(serde_json::json!({"success": false, "ok": true})),
            Some(false)
        );
        assert_eq!(v(serde_json::json!({"data": {}})), None);
    }

    #[test]
    fn self_launched_browser_note_matches_the_replacement_warning_only() {
        let note =
            |v: serde_json::Value| self_launched_browser_note(&ChromeResponse::from_value(&v));
        // The replacement note itself, top-level and inside data — the drift the
        // matcher tolerates is its parenthesized reason.
        assert!(
            note(serde_json::json!({
                "success": true,
                "warning": "This session's previous browser is gone (its browser connection was dead) and a fresh one was launched for this command"
            }))
            .is_some()
        );
        assert!(
            note(serde_json::json!({
                "ok": true,
                "data": { "warning": "This session's previous browser is gone (its browser process had exited) and a fresh one was launched for this command" }
            }))
            .is_some()
        );
        // The degraded-success warnings keep today's behaviour: a `type`
        // read-back mismatch and a key-listener note are not browser
        // replacement, and a clean envelope carries nothing at all.
        assert!(
            note(serde_json::json!({
                "success": true,
                "warning": "typed text was rewritten by the page: expected 'ab', read back 'a'"
            }))
            .is_none()
        );
        assert!(
            note(serde_json::json!({
                "success": true,
                "data": { "warning": "no key listeners on the focused element" }
            }))
            .is_none()
        );
        assert!(note(serde_json::json!({ "success": true, "data": {} })).is_none());
    }

    #[test]
    fn chrome_response_parses_structured_error_retryable() {
        let v = serde_json::json!({
            "success": false, "error": "boom", "code": "connection_failed", "retryable": true,
        });
        let resp = ChromeResponse::from_value(&v);
        assert_eq!(resp.success, Some(false));
        assert_eq!(resp.error.as_deref(), Some("boom"));
        assert_eq!(resp.code.as_deref(), Some("connection_failed"));
        assert_eq!(resp.retryable, Some(true));
        assert!(!resp.is_success());

        let v = serde_json::json!({
            "ok": false, "code": "connection_failed", "retryable": false,
        });
        let resp = ChromeResponse::from_value(&v);
        assert_eq!(resp.ok, Some(false));
        assert_eq!(resp.retryable, Some(false));
        assert!(!resp.is_success());
    }

    #[test]
    fn from_value_tolerates_unknown_keys_and_non_objects() {
        let resp = ChromeResponse::from_value(&serde_json::json!({
            "success": true, "version": 99, "unknown": [1, 2, 3],
        }));
        assert_eq!(resp.success, Some(true));
        assert!(resp.is_success());
        // Tolerates a payload with neither verdict key plus unknown keys.
        let resp = ChromeResponse::from_value(&serde_json::json!({
            "data": {"x": 1}, "future_key": true,
        }));
        assert_eq!(resp.verdict(), None);
        assert_eq!(resp.data, Some(serde_json::json!({"x": 1})));
        // A non-object / unparseable payload yields a default response.
        assert_eq!(
            ChromeResponse::from_value(&serde_json::json!("plain string")).verdict(),
            None
        );
    }

    #[test]
    fn extract_snapshot_text_handles_verified_shapes() {
        let text = |v: serde_json::Value| extract_snapshot_text(&v);
        assert_eq!(
            text(serde_json::json!("plain text")),
            Some("plain text".into())
        );
        assert_eq!(
            text(serde_json::json!({"text": "hello"})),
            Some("hello".into())
        );
        assert_eq!(text(serde_json::json!({"result": "42"})), Some("42".into()));
        assert_eq!(
            text(serde_json::json!({"content": "page"})),
            Some("page".into())
        );
        assert_eq!(
            text(serde_json::json!({"snapshot": "snap"})),
            Some("snap".into())
        );
        // Non-string fields (and absent keys) fall back in the caller.
        assert_eq!(text(serde_json::json!({"result": 42})), None);
        assert_eq!(text(serde_json::json!({})), None);
    }

    #[test]
    fn out_kind_exit_code_table() {
        assert_eq!(OutKind::Ok.exit_code(), 0);
        assert_eq!(OutKind::Empty.exit_code(), 0);
        assert_eq!(OutKind::Timeout.exit_code(), 1);
        assert_eq!(OutKind::Network.exit_code(), 1);
        assert_eq!(OutKind::Redesign.exit_code(), 1);
        assert_eq!(OutKind::NotFound.exit_code(), 1);
        assert_eq!(OutKind::Error.exit_code(), 1);
        assert_eq!(OutKind::Environment.exit_code(), 2);
        assert_eq!(OutKind::Usage.exit_code(), 3);
    }

    #[test]
    fn out_envelope_one_line_json_flattens_payload() {
        let env = OutEnvelope {
            action: "status".into(),
            ok: true,
            kind: OutKind::Ok,
            payload: serde_json::json!({ "chrome_use": "present", "relay_up": true }),
        };
        // schema first, then action/ok/kind, then the flattened payload keys.
        assert_eq!(
            env.to_json(),
            r#"{"schema":1,"action":"status","ok":true,"kind":"ok","chrome_use":"present","relay_up":true}"#
        );
    }

    /// The extract output reader reads chrome-use's envelope and tolerates the
    /// other shapes it may come in.
    #[test]
    fn extract_output_reads_the_envelope_and_tolerates_others() {
        {
            let case = "extract output reads the chrome use 15101 envelope";
            // The real 1.5.101 rows-mode shape: {extracted, count, origin, meta}.
            let data = serde_json::json!({
                "extracted": [{"a": 1}, {"a": 2}, {"a": 3}],
                "count": 3,
                "origin": "https://x",
                "meta": {},
            });
            let out = extract_output(&data, Some(2));
            assert_eq!(
                out["data"],
                serde_json::json!([{"a": 1}, {"a": 2}]),
                "case {case}: limit trims rows"
            );
            assert_eq!(out["total"], 3, "case {case}: total stays honest");
            assert_eq!(
                out["returned"], 2,
                "case {case}: returned counts the rows given back"
            );

            // Without a limit: no `returned` key.
            let out = extract_output(&data, None);
            assert_eq!(out["total"], 3, "case {case}: no limit keeps every row");
            assert!(
                out.get("returned").is_none(),
                "case {case}: no limit reports no `returned`"
            );
        }

        {
            let case = "extract output tolerates other shapes";
            // Plain array, bare object and a non-array `extracted` field each become
            // a row list; null is no rows at all.
            let shapes = [
                (
                    serde_json::json!([{"a": 1}]),
                    serde_json::json!([{"a": 1}]),
                    "plain array passes through unchanged",
                ),
                (
                    // Bare object → single-element (schemaless) row list.
                    serde_json::json!({"a": 1}),
                    serde_json::json!([{"a": 1}]),
                    "bare object is one row",
                ),
                (
                    // Non-array `extracted` field → wrapped as a single row.
                    serde_json::json!({"extracted": "weird"}),
                    serde_json::json!(["weird"]),
                    "non-array extracted is one row",
                ),
            ];
            for (input, expected, why) in shapes {
                assert_eq!(
                    extract_output(&input, None)["data"],
                    expected,
                    "case {case}: {why}"
                );
            }

            let out = extract_output(&serde_json::Value::Null, None);
            assert_eq!(
                out["data"],
                serde_json::json!([]),
                "case {case}: null is no rows"
            );
            assert_eq!(out["total"], 0, "case {case}: null has no total");
        }
    }

    #[test]
    fn expect_outcome_parses_pass_fail_and_timeout() {
        let o = expect_outcome(&serde_json::json!({"pass": true})).expect("verdict present");
        assert!(o.pass);
        assert_eq!(o.actual, None);
        assert!(!o.timed_out);

        // pass false with actual + timedOut, unknown keys ignored.
        let o = expect_outcome(&serde_json::json!({
            "pass": false, "actual": 42, "timedOut": true, "condition": "x", "kind": "count",
        }))
        .expect("verdict present");
        assert!(!o.pass);
        assert_eq!(o.actual, Some(serde_json::json!(42)));
        assert!(o.timed_out);

        // Not an expect verdict.
        assert_eq!(expect_outcome(&serde_json::json!({"success": true})), None);
        assert_eq!(expect_outcome(&serde_json::json!({"actual": 1})), None);
        assert_eq!(expect_outcome(&serde_json::json!("x")), None);
    }
    #[test]
    fn classify_call_failure_mapping() {
        // Environment signatures win (rc 2 must not be masked by page text).
        assert_eq!(
            classify_call_failure(None, "daemon may be busy or unresponsive"),
            OutKind::Environment
        );
        assert_eq!(
            classify_call_failure(Some("browser_not_launched"), "whatever"),
            OutKind::Environment
        );
        assert_eq!(
            classify_call_failure(None, "relay isn't connected"),
            OutKind::Environment
        );
        // Site-level network failures.
        assert_eq!(
            classify_call_failure(None, "net::ERR_NAME_NOT_RESOLVED"),
            OutKind::Network
        );
        assert_eq!(
            classify_call_failure(Some("connection_failed"), "connection refused"),
            OutKind::Network
        );
        // chrome-use's internal action timeout (AGENT_BROWSER_DEFAULT_TIMEOUT)
        // — kind timeout, rc 1. Chrome-side deadlines equal the declared
        // budget (the mahbot kill rides KILL_SLACK above), but the
        // classifier still catches it for every verb.
        assert_eq!(
            classify_call_failure(None, "Wait timed out after 15000ms"),
            OutKind::Timeout
        );
        assert_eq!(
            classify_call_failure(None, "waitFor timed out: .submit"),
            OutKind::Timeout
        );
        // Not-found selectors.
        assert_eq!(
            classify_call_failure(Some("element_not_found"), "whatever"),
            OutKind::NotFound
        );
        assert_eq!(
            classify_call_failure(None, "Element not found"),
            OutKind::NotFound
        );
        // Everything else is a generic step failure.
        assert_eq!(
            classify_call_failure(None, "some random issue"),
            OutKind::Error
        );
        assert_eq!(
            classify_call_failure(Some("other_code"), "some issue"),
            OutKind::Error
        );
    }

    /// A network error's code is read from either form and bucketed into dns, refused or other.
    #[test]
    fn net_error_extraction_buckets_both_code_forms() {
        {
            let case = "net error code extraction canonicalizes both forms";
            // The net::ERR_ prefix form (chrome-use errorText) and the bare
            // ERR_*/DNS_* form (div.error-code) canonicalize to the bare token.
            for (text, expected) in [
                (
                    "Navigation failed: net::ERR_NAME_NOT_RESOLVED",
                    Some("ERR_NAME_NOT_RESOLVED"),
                ),
                ("ERR_CONNECTION_REFUSED", Some("ERR_CONNECTION_REFUSED")),
                (
                    "DNS_PROBE_FINISHED_NXDOMAIN",
                    Some("DNS_PROBE_FINISHED_NXDOMAIN"),
                ),
                (
                    "error: net::ERR_CONNECTION_TIMED_OUT — retry",
                    Some("ERR_CONNECTION_TIMED_OUT"),
                ),
                ("connection refused by peer", None),
                ("", None),
            ] {
                assert_eq!(
                    extract_net_error_code(text).as_deref(),
                    expected,
                    "case {case}: {text:?}"
                );
            }
        }

        {
            let case = "net error phrases bucket dns refused other";
            for (code, expected) in [
                // ERR_NAME_NOT_RESOLVED is the DNS-failure code surfaced via CDP
                // errorText despite the ERR_ prefix.
                ("DNS_PROBE_FINISHED_NXDOMAIN", "DNS resolution failure"),
                ("ERR_NAME_NOT_RESOLVED", "DNS resolution failure"),
                ("ERR_CONNECTION_REFUSED", "connection refused"),
                (
                    "ERR_CONNECTION_RESET",
                    "site unreachable or connection failed",
                ),
            ] {
                assert_eq!(net_error_phrase(code), expected, "case {case}: {code}");
            }
        }
    }

    #[test]
    fn error_page_probe_parses_verdict_shapes() {
        let resp = |data: Value| ChromeResponse {
            data: Some(data),
            ..Default::default()
        };
        // The usual shape: the JSON verdict string inside the eval wrapper.
        let clean = resp(json!({ "origin": "x", "result": r#"{"err": false, "code": ""}"# }));
        assert_eq!(
            parse_error_page_probe(&clean),
            Some(ErrorPageProbe {
                is_error_page: false,
                code: None
            })
        );
        let failed = resp(
            json!({ "origin": "x", "result": r#"{"err": true, "code": "ERR_CONNECTION_REFUSED"}"# }),
        );
        assert_eq!(
            parse_error_page_probe(&failed),
            Some(ErrorPageProbe {
                is_error_page: true,
                code: Some("ERR_CONNECTION_REFUSED".into())
            })
        );
        // An error page whose code did not render in time → generic fallback.
        let no_code = resp(json!({ "origin": "x", "result": r#"{"err": true, "code": ""}"# }));
        assert_eq!(
            parse_error_page_probe(&no_code),
            Some(ErrorPageProbe {
                is_error_page: true,
                code: None
            })
        );
        // A bare (wrapper-less) string verdict.
        let bare = resp(json!(
            r#"{"err": true, "code": "DNS_PROBE_FINISHED_NXDOMAIN"}"#
        ));
        assert_eq!(
            parse_error_page_probe(&bare),
            Some(ErrorPageProbe {
                is_error_page: true,
                code: Some("DNS_PROBE_FINISHED_NXDOMAIN".into())
            })
        );
        // An already-parsed verdict object.
        let object = resp(json!({ "err": true, "code": "ERR_UNSAFE_PORT" }));
        assert_eq!(
            parse_error_page_probe(&object),
            Some(ErrorPageProbe {
                is_error_page: true,
                code: Some("ERR_UNSAFE_PORT".into())
            })
        );
        // Lenient fallback: an unparseable text carrying a net error token
        // still identifies an error page.
        let token_text = resp(json!("net::ERR_INTERNET_DISCONNECTED"));
        assert_eq!(
            parse_error_page_probe(&token_text),
            Some(ErrorPageProbe {
                is_error_page: true,
                code: Some("ERR_INTERNET_DISCONNECTED".into())
            })
        );
        // Inconclusive shapes → None (the probe passes, never a failure).
        assert_eq!(parse_error_page_probe(&resp(json!(null))), None);
        assert_eq!(parse_error_page_probe(&resp(json!(""))), None);
        assert_eq!(parse_error_page_probe(&resp(json!("garbage text"))), None);
        assert_eq!(parse_error_page_probe(&resp(json!(true))), None);
    }

    /// A timeout message loses the canned stale-relay hint, and a message that
    /// is not a timeout is left alone.
    #[test]
    fn sanitize_timeout_message_strips_only_the_stale_relay_hint() {
        {
            let case = "sanitize timeout message strips stale relay hint";
            // Timeout-classified message with the canned hint → hint stripped
            // (including the separator period before the hint sentence), honest
            // prefix ("Wait timed out after 15000ms") kept. A Timeout message
            // without the marker is unchanged.
            for (input, expected) in [
                (
                    "Wait timed out after 15000ms. Hint: the session's browser connection \
                     is unresponsive (likely a stale relay/service-worker mid-session). Reconnect \
                     with `connect`, or close the session and reopen it.",
                    "Wait timed out after 15000ms",
                ),
                (
                    "Wait timed out after 19000ms",
                    "Wait timed out after 19000ms",
                ),
            ] {
                assert_eq!(
                    sanitize_timeout_message(OutKind::Timeout, input),
                    expected,
                    "case {case}: {input:?}"
                );
            }
        }

        {
            let case = "sanitize timeout message leaves non timeout alone";
            // The hint with NO timeout phrasing → not Timeout, so untouched; an
            // Environment-classified relay message and an Error-classified message
            // carrying the hint are left verbatim too.
            let hint = "Hint: the session's browser connection is unresponsive (likely a stale \
                        relay/service-worker mid-session). Reconnect with `connect`, or close the session \
                        and reopen it.";
            let untouched = [
                (OutKind::Error, hint.to_string()),
                (OutKind::Environment, "relay isn't connected".to_string()),
                (OutKind::Error, format!("some other failure\n{hint}")),
            ];
            for (kind, message) in untouched {
                assert_eq!(
                    sanitize_timeout_message(kind, &message),
                    message,
                    "case {case}: {kind:?}"
                );
            }
        }
    }

    /// Only a phrased wait/expect timeout earns its action's remediation note: an
    /// empty message is the mahbot-side deadline kill and keeps its own factual
    /// text, and every other kind or action is a no-op.
    #[test]
    fn with_condition_timeout_note_appends_by_action() {
        // Phrased wait timeout → wait note; expect → expect note.
        let mut m = String::from("Wait timed out after 15000ms");
        with_condition_timeout_note("wait", OutKind::Timeout, &mut m);
        assert_eq!(
            m,
            format!("Wait timed out after 15000ms{WAIT_TIMEOUT_NOTE}"),
            "wait gets the target remediation"
        );
        let mut m = String::from("waitFor timed out: .submit");
        with_condition_timeout_note("expect", OutKind::Timeout, &mut m);
        assert_eq!(
            m,
            format!("waitFor timed out: .submit{EXPECT_TIMEOUT_NOTE}"),
            "expect gets the condition remediation"
        );

        // Empty message (mahbot-side deadline kill) → untouched, so the deadline
        // kill keeps its own factual error text and never gets the remediation.
        let mut m = String::new();
        with_condition_timeout_note("wait", OutKind::Timeout, &mut m);
        assert!(m.is_empty(), "an empty message is never given the note");

        // Non-Timeout kinds and other actions are never touched.
        let mut m = String::from("some failure");
        with_condition_timeout_note("wait", OutKind::Error, &mut m);
        with_condition_timeout_note("eval", OutKind::Timeout, &mut m);
        assert_eq!(m, "some failure", "every other kind or action is a no-op");
    }

    #[test]
    fn session_unresponsive_error_signature_detected() {
        for msg in [
            "session unresponsive: no response within 45s",
            "CDP session is unresponsive after attaching (Connection reset).",
        ] {
            assert!(is_session_unresponsive_error(msg), "should detect: {msg}");
        }
        // The daemon-busy phrasing is a DaemonWedge text, not a session wedge.
        assert!(!is_session_unresponsive_error(
            "daemon may be busy or unresponsive"
        ));
    }

    #[test]
    fn unreachable_tab_signature_and_message_are_stable() {
        for msg in [
            "the tab this session was driving can no longer be resolved (it was closed, or a flaky relay dropped it)",
            "the tab this command was driving is gone — it may have been closed, or the relay lost it",
            "this session owns no resolvable tab in its group. Refusing to run on a tab this session does not drive",
            "no attached tab ...",
            "unknown sessionId ...",
            "stale sessionId ... its tab is gone",
        ] {
            assert!(is_unreachable_tab_error(msg), "should detect: {msg}");
        }
        // The recoverable OAuth/SSO retarget is not a tab-attach problem.
        assert!(!is_unreachable_tab_error(
            "the tab this command was driving is gone — it navigated across processes"
        ));
        assert!(!is_unreachable_tab_error(
            "Auto-launch failed: Could not drive your Chrome through the ab-connect extension."
        ));
        assert!(!is_unreachable_tab_error(
            "Failed to read: Resource temporarily unavailable (os error 35)"
        ));
        assert!(!is_unreachable_tab_error(
            "chrome-use error: Element not found"
        ));
        let m = unreachable_tab_message("this session owns no resolvable tab");
        assert!(m.ends_with("the chrome daemon itself is healthy)."));
        assert!(m.contains("the product closes a session's tabs itself"));
        assert!(!m.contains("close the leftover tab in Chrome"));
    }

    #[test]
    fn unreached_browser_error_covers_structural_classes_but_not_a_refusal() {
        for msg in [
            "relay isn't connected",
            "daemon may be busy or unresponsive",
            "session unresponsive: no response within 45s",
            "Browser not launched",
            "this action requires ab-connect",
            "tabs.remove only works over the extension relay",
            "event stream closed before the call was answered",
            "command timed out after 30000ms",
            "stale target: the tab has moved",
            "Target closed",
        ] {
            assert!(unreached_browser_error(msg), "should detect: {msg}");
        }
        // The extension's own ownership refusal IS the browser answering, never an unreached class.
        assert!(!unreached_browser_error(
            "call: tabs.remove refused — tab 7 is not owned by this relay \
             (agent-created or adopted tabs only)"
        ));
        assert!(!unreached_browser_error("Element not found"));
    }
}
