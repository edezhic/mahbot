//! One place for the chrome-use response envelope parsing, shared by the
//! interactive `chrome` tool and the `mahbot chrome` CLI.
//!
//! chrome-use answers `--json` commands with either the classic `success` key
//! or the newer `ok` key. [`ChromeResponse`] is the single structural parser
//! every frontend shares — [`ChromeResponse::verdict`] applies the shared
//! failure-precedence logic so the two paths cannot drift. The `mahbot
//! chrome` stdout output contract ([`SCHEMA_VERSION`], [`OutKind`],
//! [`OutEnvelope`]) lives here too.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::LazyLock;

/// Response from chrome-use `--json` commands — the ONE structural envelope
/// parser every chrome frontend shares.
///
/// There is no schema-version marker in chrome-use, and the envelope even
/// varies within one binary (`session list` answers `ok:true`, everything
/// else `success:true`), so tolerance is structural: every field is
/// [`Option`], unknown keys are captured in [`extra`](Self::extra).
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChromeResponse {
    /// Classic `success` envelope key — absent on newer `ok`-keyed envelopes.
    #[serde(default)]
    pub(crate) success: Option<bool>,
    /// Latest chrome-use envelopes replace `success` with `ok` on some commands
    /// (e.g. `session list --json`).
    #[serde(default)]
    pub(crate) ok: Option<bool>,
    pub(crate) data: Option<Value>,
    pub(crate) error: Option<String>,
    /// Stable error-envelope code (v1.5.78+) — present on structured errors.
    pub(crate) code: Option<String>,
    /// `retryable` flag on structured error envelopes (v1.5.101 emits it on
    /// every `{success:false,…}`) — the interactive tool surfaces it to the
    /// agent as a retry hint.
    #[serde(default)]
    pub(crate) retryable: Option<bool>,
    /// Any other top-level envelope keys, captured so chrome-use's degraded-
    /// success `warning` field is visible wherever it puts it
    /// ([`chrome_use_warning`]).
    #[serde(flatten)]
    pub(crate) extra: serde_json::Map<String, Value>,
}

impl ChromeResponse {
    /// Success gate accepting both the classic `success: true` envelope and
    /// the latest `ok: true` one — delegates to the shared [`envelope_success`]
    /// predicate (failure precedence).
    pub(crate) fn is_success(&self) -> bool {
        envelope_success(self.success, self.ok)
    }

    /// Tri-state verdict of a chrome-use JSON envelope: `Some(true)` when
    /// either `success` or `ok` reports success, `Some(false)` when either
    /// reports failure, `None` when the payload carries no verdict key
    /// (callers own their default: strict / tolerance-first / error-first).
    pub(crate) fn verdict(&self) -> Option<bool> {
        if self.success.is_none() && self.ok.is_none() {
            return None;
        }
        Some(self.is_success())
    }

    /// The single `Value`-path parse entry: unknown keys are captured in
    /// `extra` and any non-object / unparseable payload yields a default
    /// response (all fields [`None`]).
    #[must_use]
    pub(crate) fn from_value(v: &Value) -> Self {
        serde_json::from_value(v.clone()).unwrap_or_default()
    }
}

/// chrome-use's `warning` field — the degraded-success signal for `type`
/// (the page rewrote or filtered the typed text) and `press` (the key reached
/// an element with no key listeners, or provably went nowhere). It rides
/// inside `data` (the command's own warning) or at the envelope top level
/// (the flattened `extra` map — chrome-use's `Response` struct carries a
/// top-level `warning` for browser-replacement and settle notes, which also
/// mean the typed input may not be where we think). chrome-use emits it ONLY
/// on a degraded exit-0 success — never on a clean one — so its presence
/// means "the action may not have taken effect".
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

/// Core envelope-success predicate with failure precedence: an explicit
/// `false` on either key loses over a contradicting success key (conservative
/// — the error text surfaces), and a payload with neither key is not a
/// success. Backs [`ChromeResponse::is_success`] and
/// [`ChromeResponse::verdict`].
fn envelope_success(success: Option<bool>, ok: Option<bool>) -> bool {
    !(success == Some(false) || ok == Some(false)) && (success == Some(true) || ok == Some(true))
}

/// Extract textual content from a chrome-use snapshot response `data` field.
///
/// chrome-use can return the snapshot as:
/// - A plain string (via `snapshot -c`)
/// - An object with a `text` field (via `get_text`)
/// - An object with a `result` field (via `eval`)
/// - An object with a `content` field (via `get_text`)
/// - An object with `origin`, `refs`, and `snapshot` fields (via `snapshot --json`)
///
/// Returns `None` if none of these shapes match (the caller falls back).
pub(crate) fn extract_snapshot_text(data: &serde_json::Value) -> Option<String> {
    data.as_str().map(String::from).or_else(|| {
        ["text", "result", "content", "snapshot"]
            .iter()
            .find_map(|key| data.get(*key).and_then(Value::as_str).map(String::from))
    })
}

/// Unwrap the actual JS result from an eval response. chrome-use wraps
/// `eval` output as `data = {origin, result}` — an object carrying BOTH keys
/// is treated as that wrapper; every other shape (a bare value, a plain
/// string, a page result that merely has a `result` field) passes through.
pub(crate) fn eval_result(resp: &ChromeResponse) -> Option<&Value> {
    resp.data
        .as_ref()
        .map(|d| match (d.get("result"), d.get("origin")) {
            (Some(result), Some(_)) => result,
            _ => d,
        })
}

/// The ONE error-page probe eval, shared by the CLI `open` action and the
/// interactive tool: always a JSON verdict string (robust against textified
/// delivery) reporting whether the tab committed to a Chrome error page and,
/// best-effort, the net error token Chrome renders in the error page's
/// `div.error-code` (empty until the neterror script renders it, and on
/// template/locale drift — callers must keep the generic fallback).
pub(crate) const ERROR_PAGE_PROBE_JS: &str = "JSON.stringify({err: location.protocol === 'chrome-error:', code: (document.querySelector('.error-code') || {}).textContent || ''})";

/// Parsed [`ERROR_PAGE_PROBE_JS`] verdict.
#[derive(Debug, PartialEq)]
pub(crate) struct ErrorPageProbe {
    /// The tab conclusively sits on a Chrome error page.
    pub(crate) is_error_page: bool,
    /// The net error token (`ERR_*` / `DNS_*`) read from `div.error-code`,
    /// when one rendered and is recognizable.
    pub(crate) code: Option<String>,
}

/// Parse an error-page probe response. `None` = inconclusive (callers treat
/// the probe as passed, never as a network failure). The verdict arrives as
/// the eval's JSON string — possibly textified — or an already-parsed object.
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

/// Extract a non-negative element count from a count-eval response — the eval
/// result arrives as a JSON number or its text form.
pub(crate) fn eval_count(resp: &ChromeResponse) -> Option<u64> {
    let data = eval_result(resp)?;
    data.as_u64()
        .or_else(|| data.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
}

/// Trim `rows` to `limit` (mahbot-side) while reporting the honest `total`.
/// Returns `(trimmed, total, returned)`.
fn trim_extract_rows(rows: Vec<Value>, limit: Option<usize>) -> (Vec<Value>, usize, usize) {
    let total = rows.len();
    let returned = limit.map_or(total, |l| l.min(total));
    let trimmed: Vec<Value> = rows.into_iter().take(returned).collect();
    (trimmed, total, returned)
}

/// Rows from a chrome-use `extract --json` response `data` field.
///
/// chrome-use 1.5.101 answers rows-mode extract with
/// `data = {extracted: [...], count, origin, meta}` (a `diagnostic` string is
/// added on 0-match — unreachable through the count-gated frontends).
/// Tolerance: a plain array passes through; a bare object is a single-object
/// (schemaless) extract and becomes a one-element row list.
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

/// The ONE extract output contract for both frontends: read the rows from a
/// chrome-use extract response `data` value and shape the payload — `data`
/// carries the (possibly `limit`-trimmed) rows, `total` the honest full
/// count, and `returned` appears only when a `limit` was applied.
#[must_use]
pub(crate) fn extract_output(data: &Value, limit: Option<usize>) -> Value {
    let (trimmed, total, returned) = trim_extract_rows(extract_rows(data), limit);
    let mut payload = json!({ "data": trimmed, "total": total });
    if limit.is_some() {
        payload["returned"] = json!(returned);
    }
    payload
}

/// A parsed `expect` verdict from a chrome-use `expect --json` response
/// `data` object. `pass` is the authoritative tri-state anchor: `None` from
/// [`expect_outcome`] means the payload was not an expect verdict at all.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpectOutcome {
    pub(crate) pass: bool,
    /// The observed value chrome-use evaluated (element presence object,
    /// count, text, URL string, …) — `None` when absent.
    pub(crate) actual: Option<Value>,
    /// chrome-use sets `timedOut: true` when the condition never held within
    /// the deadline. mahbot never passes --no-wait, so a false verdict always
    /// arrives folded into `timedOut` in practice; a bare `pass:false` is
    /// defensive against a future chrome-use grammar change.
    pub(crate) timed_out: bool,
}

/// Parse the `expect` verdict out of a chrome-use response `data` value.
/// Returns `None` when the payload carries no boolean `pass` key (not an
/// expect verdict — the caller classifies it as a generic error).
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

/// Failure/success classification of one CLI step — drives the emitted
/// `kind` and the exit code (site problems are rc 1 and schedulable;
/// environment problems are rc 2 and need a human).
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

    /// Process exit code for this kind: site/data problems are rc 1 and
    /// retryable; environment problems are rc 2 (fix the environment, don't
    /// blind-retry); usage is rc 3.
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

// ── Failure signatures & classification ────────────────────────────────
// These live here (not in chrome_daemon) so the failure-interpretation layer
// stays dependency-free: chrome_daemon imports them back, and both frontends
// classify through this module without any frontend-to-frontend coupling.

/// Per-session wedge signature: the session's CDP attach is unresponsive and
/// every command on that named session times out while the daemon and relay
/// may be perfectly healthy. The chrome_daemon watchdog reaches these texts
/// through [`is_daemon_unavailable_error`] (they are DaemonWedge signals and
/// MUST keep triggering its auto-recovery); the CLI uses this matcher
/// directly to attach the wedge-specific remediation hint (the OutKind is
/// Environment for both, so classification is unchanged).
pub(crate) fn is_session_unresponsive_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("session unresponsive")
        || lower.contains("cdp session is unresponsive after attaching")
}

/// Detect the daemon-unavailable signature chrome-use produces when its
/// background daemon is dead or wedged: the CLI hangs in its own 5-retry loop
/// (EAGAIN / "Resource temporarily unavailable") and eventually reports
/// "daemon may be busy or unresponsive". Also covers the 1.5.8x-era texts
/// (stuck-daemon auto-stop, disappeared daemon endpoint, failed auto-launch)
/// and — via the [`is_session_unresponsive_error`] delegation — the
/// session-wedge texts, which are DaemonWedge signals for the watchdog's
/// auto-recovery.
pub(crate) fn is_daemon_unavailable_error(msg: &str) -> bool {
    // The session-wedge texts are also DaemonWedge signals — the watchdog's
    // auto-recovery cause folds them in (see [`is_session_unresponsive_error`]).
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
        // Colon form only — "failed to connect to <host>" is a page-level
        // navigation failure, not a daemon socket problem.
        || lower.contains("failed to connect:")
}

/// Relay-side failure signature — the daemon is alive but cannot drive Chrome
/// through the extension relay. Distinct from a daemon wedge (restart clears a
/// wedge; a relay drop needs the extension to republish, then self-heals).
pub(crate) fn is_relay_unavailable_error(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("relay isn't connected")
        || lower.contains("relay is not")
        || lower.contains("relay dropped")
        || lower.contains("relay down")
        || lower.contains("could not drive your chrome")
}

/// Stable error-envelope `code` values (v1.5.78+) that are unambiguously
/// daemon-side. The coarse `connection_failed` code is shared with page-level
/// navigation failures (the CLI classifies any "connection" text as such), so
/// the message-text matcher stays the source of truth for those.
pub(crate) fn is_daemon_unavailable_code(code: Option<&str>) -> bool {
    matches!(code, Some("browser_not_launched"))
}

/// Error signatures of a leftover tab the daemon can no longer re-drive: its
/// binding went stale (relay blip, kill during an outage) while the extension
/// keeps the attach. The extension never re-attaches `about:` URLs, so only
/// closing the tab by hand unblocks the session. An orphan the extension fully
/// dropped (service-worker restart) never produces these. Real chrome calls
/// hitting this state fail fast with the same guidance without marking the
/// daemon unhealthy — see [`unreachable_tab_message`]. The "or the relay lost
/// it" variant is a permanent orphan (the relay dropped the attach);
/// "navigated across processes" is a recoverable OAuth/SSO retarget and must
/// stay OUT.
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

/// Actionable error for a real call that hit an orphaned tab: the daemon and
/// relay are up, only the session's tab is unreachable (the extension never
/// re-attaches about:blank tabs). Fail fast with hand-close guidance and do
/// NOT mark the daemon unhealthy — recovery cannot fix a Chrome-side orphan.
pub(crate) fn unreachable_tab_message(error: &str) -> String {
    format!(
        "{error}. The chrome-use extension lost its debugger attach to this tab and never \
         re-attaches about:blank tabs — close the leftover tab in Chrome to unblock this \
         session (the chrome daemon itself is healthy)."
    )
}

/// Classify a chrome-use step failure into an [`OutKind`]. Environment
/// signatures win because rc 2 must not be masked by a page-level wrapper
/// text: a relay/daemon/Chrome-launch problem is a CLI environment failure
/// even when the CLI wraps the message in a "page failed" envelope.
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
    // chrome-use's internal action timeout (AGENT_BROWSER_DEFAULT_TIMEOUT).
    // Chrome-side deadlines equal the declared budget, and the mahbot-side
    // kill rides DEADLINE_SLACK above them, so chrome-use's honest phrased
    // timeouts normally surface first; the classifier still catches them for
    // every verb. Matched
    // by its specific phrasings ("Wait timed out after Nms", "waitFor timed
    // out: …") rather than a bare "timed out" substring, so an unrecognized
    // connect-timeout message still classifies as Error — same exit code (1),
    // honest kind label.
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

/// Canonical net error token extracted from chrome-side text. Both forms
/// chrome-use surfaces — the `net::ERR_*` prefix form on CDP `errorText` and
/// the bare `ERR_*`/`DNS_*` form Chrome renders in the error page's
/// `div.error-code` — canonicalize to the bare token (e.g.
/// `ERR_NAME_NOT_RESOLVED`, `DNS_PROBE_FINISHED_NXDOMAIN`).
pub(crate) fn extract_net_error_code(text: &str) -> Option<String> {
    static NET_ERROR_CODE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?:net::)?\b(ERR_[A-Z0-9_]+|DNS_[A-Z0-9_]+)").expect("valid net error regex")
    });
    NET_ERROR_CODE
        .captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// Coarse cause phrase for a net error token: DNS failures, refused
/// connections, everything else (timed out/reset/unsafe port/disconnected…)
/// shares one honest "unreachable or failed" bucket — the raw token is the
/// precise signal for the caller. `ERR_NAME_NOT_RESOLVED` counts as DNS: it
/// is the canonical DNS-failure code chrome-use surfaces via CDP errorText
/// (the error page itself renders the `DNS_*`-prefixed token).
pub(crate) fn net_error_phrase(code: &str) -> &'static str {
    if code.starts_with("DNS_") || code == "ERR_NAME_NOT_RESOLVED" {
        "DNS resolution failure"
    } else if code == "ERR_CONNECTION_REFUSED" {
        "connection refused"
    } else {
        "site unreachable or connection failed"
    }
}

/// Start of the canned remediation hint chrome-use (external binary) appends
/// to some timeout failures. Everything from the first occurrence of this
/// marker to the end of the message is stripped by
/// [`sanitize_timeout_message`].
const STALE_RELAY_HINT_MARKER: &str = "Hint: the session's browser connection is unresponsive";

/// chrome-use (external binary) appends a canned remediation hint to some
/// timeout failures: "Hint: the session's browser connection is unresponsive
/// (likely a stale relay/service-worker mid-session). Reconnect with
/// `connect`, or close the session and reopen it." That hint is wrong in
/// mahbot's surface — there is no `connect` verb here and the connection is
/// usually healthy (the condition simply never appeared) — so both frontends
/// strip it from Timeout-classified messages. Environment-classified relay/
/// daemon diagnostics are never rewritten: genuine connection-failure
/// guidance must pass through untouched. The wording is chrome-use's; if a
/// release rephrases it, the strip degrades to a no-op (acceptable). The
/// separator period before the hint sentence is dropped too, so appended
/// remediation reads "…15000ms — …" rather than "…15000ms. — …".
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

/// Remediation appended to `wait` condition timeouts in BOTH frontends (after
/// the honest chrome-use text) — one source so the CLI and the interactive
/// tool never drift.
pub(crate) const WAIT_TIMEOUT_NOTE: &str = " — the target never appeared within the deadline; consider verifying the \
     selector/text/URL or allowing more time";

/// Same for `expect` (whose cause text already names the deadline).
pub(crate) const EXPECT_TIMEOUT_NOTE: &str =
    " — consider verifying the condition or allowing more time";

/// Append the action's condition-timeout remediation note to `message` when
/// `kind` is Timeout and `message` is chrome-use's own phrased text — an empty
/// message is the mahbot-side deadline kill, which carries its own factual
/// error text and never gets the remediation note. No-op for every other
/// action.
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

/// The one-line stdout envelope. `payload` must be a JSON object; it is
/// flattened after `kind`.
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

        // Failure precedence — an explicit false on either key loses over a
        // contradicting success key.
        let mixed: ChromeResponse =
            serde_json::from_str(r#"{"success":false,"ok":true}"#).expect("tolerant deserialize");
        assert!(!mixed.is_success());
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
    fn out_kind_as_str_matches_contract() {
        for (kind, s) in [
            (OutKind::Ok, "ok"),
            (OutKind::Empty, "empty"),
            (OutKind::Timeout, "timeout"),
            (OutKind::Network, "network"),
            (OutKind::Redesign, "redesign"),
            (OutKind::NotFound, "not-found"),
            (OutKind::Error, "error"),
            (OutKind::Environment, "environment"),
            (OutKind::Usage, "usage"),
        ] {
            assert_eq!(kind.as_str(), s);
        }
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

    #[test]
    fn extract_output_reads_the_chrome_use_15101_envelope() {
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
            "limit trims rows"
        );
        assert_eq!(out["total"], 3, "total stays honest");
        assert_eq!(out["returned"], 2);

        // Without a limit: no `returned` key.
        let out = extract_output(&data, None);
        assert_eq!(out["total"], 3);
        assert!(out.get("returned").is_none());
    }

    #[test]
    fn extract_output_tolerates_other_shapes() {
        // Plain array passes through unchanged.
        let arr = serde_json::json!([{"a": 1}]);
        assert_eq!(
            extract_output(&arr, None)["data"],
            serde_json::json!([{"a": 1}])
        );

        // Bare object → single-element (schemaless) row list.
        let obj = serde_json::json!({"a": 1});
        assert_eq!(
            extract_output(&obj, None)["data"],
            serde_json::json!([{"a": 1}])
        );

        // Non-array `extracted` field → wrapped as a single row.
        let weird = serde_json::json!({"extracted": "weird"});
        assert_eq!(
            extract_output(&weird, None)["data"],
            serde_json::json!(["weird"])
        );

        // Null → empty rows, total 0.
        let out = extract_output(&serde_json::Value::Null, None);
        assert_eq!(out["data"], serde_json::json!([]));
        assert_eq!(out["total"], 0);
    }

    #[test]
    fn expect_outcome_parses_pass_fail_and_timeout() {
        // pass true, no timeout, no actual.
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
        // budget (the mahbot kill rides DEADLINE_SLACK above), but the
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

    #[test]
    fn net_error_code_extraction_canonicalizes_both_forms() {
        // The net::ERR_ prefix form (chrome-use errorText) and the bare
        // ERR_*/DNS_* form (div.error-code) canonicalize to the bare token.
        assert_eq!(
            extract_net_error_code("Navigation failed: net::ERR_NAME_NOT_RESOLVED"),
            Some("ERR_NAME_NOT_RESOLVED".into())
        );
        assert_eq!(
            extract_net_error_code("ERR_CONNECTION_REFUSED"),
            Some("ERR_CONNECTION_REFUSED".into())
        );
        assert_eq!(
            extract_net_error_code("DNS_PROBE_FINISHED_NXDOMAIN"),
            Some("DNS_PROBE_FINISHED_NXDOMAIN".into())
        );
        assert_eq!(
            extract_net_error_code("error: net::ERR_CONNECTION_TIMED_OUT — retry"),
            Some("ERR_CONNECTION_TIMED_OUT".into())
        );
        assert_eq!(extract_net_error_code("connection refused by peer"), None);
        assert_eq!(extract_net_error_code(""), None);
    }

    #[test]
    fn net_error_phrases_bucket_dns_refused_other() {
        assert_eq!(
            net_error_phrase("DNS_PROBE_FINISHED_NXDOMAIN"),
            "DNS resolution failure"
        );
        // ERR_NAME_NOT_RESOLVED is the DNS-failure code surfaced via CDP
        // errorText despite the ERR_ prefix.
        assert_eq!(
            net_error_phrase("ERR_NAME_NOT_RESOLVED"),
            "DNS resolution failure"
        );
        assert_eq!(
            net_error_phrase("ERR_CONNECTION_REFUSED"),
            "connection refused"
        );
        assert_eq!(
            net_error_phrase("ERR_CONNECTION_RESET"),
            "site unreachable or connection failed"
        );
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

    #[test]
    fn sanitize_timeout_message_strips_stale_relay_hint() {
        // Timeout-classified message with the canned hint → hint stripped
        // (including the separator period before the hint sentence), honest
        // prefix ("Wait timed out after 15000ms") kept.
        assert_eq!(
            sanitize_timeout_message(
                OutKind::Timeout,
                "Wait timed out after 15000ms. Hint: the session's browser connection \
                 is unresponsive (likely a stale relay/service-worker mid-session). Reconnect \
                 with `connect`, or close the session and reopen it."
            ),
            "Wait timed out after 15000ms"
        );
        // A Timeout message without the marker is unchanged.
        assert_eq!(
            sanitize_timeout_message(OutKind::Timeout, "Wait timed out after 19000ms"),
            "Wait timed out after 19000ms"
        );
    }

    #[test]
    fn sanitize_timeout_message_leaves_non_timeout_alone() {
        // The hint with NO timeout phrasing → not Timeout, so untouched.
        let hint = "Hint: the session's browser connection is unresponsive (likely a stale \
             relay/service-worker mid-session). Reconnect with `connect`, or close the session \
             and reopen it.";
        assert_eq!(sanitize_timeout_message(OutKind::Error, hint), hint);
        // An Environment-classified relay message passes through untouched.
        let relay = "relay isn't connected";
        assert_eq!(sanitize_timeout_message(OutKind::Environment, relay), relay);
        // An Error-classified message with the hint keeps it verbatim.
        let other = format!("some other failure\n{hint}");
        assert_eq!(sanitize_timeout_message(OutKind::Error, &other), other);
    }

    #[test]
    fn with_condition_timeout_note_appends_by_action() {
        // Phrased wait timeout → wait note; expect → expect note.
        let mut m = String::from("Wait timed out after 15000ms");
        with_condition_timeout_note("wait", OutKind::Timeout, &mut m);
        assert_eq!(
            m,
            format!("Wait timed out after 15000ms{WAIT_TIMEOUT_NOTE}")
        );
        let mut m = String::from("waitFor timed out: .submit");
        with_condition_timeout_note("expect", OutKind::Timeout, &mut m);
        assert_eq!(
            m,
            format!("waitFor timed out: .submit{EXPECT_TIMEOUT_NOTE}")
        );
        // Empty message (mahbot-side deadline kill) → untouched, so the
        // deadline kill keeps its own factual error text.
        let mut m = String::new();
        with_condition_timeout_note("wait", OutKind::Timeout, &mut m);
        assert!(m.is_empty());
        // Non-Timeout kinds and other actions are never touched.
        let mut m = String::from("some failure");
        with_condition_timeout_note("wait", OutKind::Error, &mut m);
        with_condition_timeout_note("eval", OutKind::Timeout, &mut m);
        assert_eq!(m, "some failure");
    }

    #[test]
    fn session_unresponsive_error_signature_detected() {
        for msg in [
            "session unresponsive: no response within 45s",
            "CDP session is unresponsive after attaching (Connection reset).",
        ] {
            assert!(is_session_unresponsive_error(msg), "should detect: {msg}");
        }
        // The daemon-busy phrasing is a DaemonWedge text, not a session wedge —
        // the CLI must not attach the session hint to it.
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
        // The recoverable OAuth/SSO retarget and daemon/page-level failures
        // are not tab-attach problems.
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
        assert!(m.contains("close the leftover tab in Chrome"));
    }
}
