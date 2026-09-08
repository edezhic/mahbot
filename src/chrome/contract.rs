//! One place for the chrome-use response envelope parsing, shared by the
//! interactive `chrome` tool and the `mahbot chrome` CLI.
//!
//! chrome-use answers `--json` commands with either the classic `success` key
//! or the newer `ok` key. [`ChromeResponse`] is the single structural parser
//! every frontend shares — [`ChromeResponse::verdict`] applies the shared
//! failure-precedence logic so the two paths cannot drift. The `mahbot
//! chrome` stdout output contract ([`SCHEMA_VERSION`], [`OutKind`],
//! [`OutEnvelope`]) lives here too.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
}
