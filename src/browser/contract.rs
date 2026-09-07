//! One place for the chrome-use response envelope parsing, shared by the
//! interactive `browser` tool and the `mahbot browser` CLI.
//!
//! chrome-use answers `--json` commands with either the classic `success` key
//! or the newer `ok` key. [`BrowserResponse`] is the single structural parser
//! every frontend shares — [`BrowserResponse::verdict`] applies the shared
//! failure-precedence logic so the two paths cannot drift. The `mahbot
//! browser` stdout output contract ([`SCHEMA_VERSION`], [`OutKind`],
//! [`OutEnvelope`]) lives here too.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response from chrome-use `--json` commands — the ONE structural envelope
/// parser every browser frontend shares.
///
/// There is no schema-version marker in chrome-use, and the envelope even
/// varies within one binary (`session list` answers `ok:true`, everything
/// else `success:true`), so tolerance is structural: every field is
/// [`Option`], unknown keys are ignored.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct BrowserResponse {
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
}

impl BrowserResponse {
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

    /// The single `Value`-path parse entry: unknown keys are dropped and any
    /// non-object / unparseable payload yields a default response (all fields
    /// [`None`]).
    #[must_use]
    pub(crate) fn from_value(v: &Value) -> Self {
        serde_json::from_value(v.clone()).unwrap_or_default()
    }
}

/// Core envelope-success predicate with failure precedence: an explicit
/// `false` on either key loses over a contradicting success key (conservative
/// — the error text surfaces), and a payload with neither key is not a
/// success. Backs [`BrowserResponse::is_success`] and
/// [`BrowserResponse::verdict`].
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
/// - An object with `origin`, `refs`, and `snapshot` fields (via `open` auto-snapshot)
///
/// Returns `None` if none of these shapes match (the caller falls back).
pub(crate) fn extract_snapshot_text(data: &serde_json::Value) -> Option<String> {
    data.as_str().map(String::from).or_else(|| {
        ["text", "result", "content", "snapshot"]
            .iter()
            .find_map(|key| data.get(*key).and_then(Value::as_str).map(String::from))
    })
}

/// `mahbot browser` stdout envelope schema version.
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
    fn browser_response_accepts_tolerant_envelopes() {
        let ok: BrowserResponse =
            serde_json::from_str(r#"{"ok":true,"data":{}}"#).expect("tolerant deserialize");
        assert!(ok.is_success());

        let success: BrowserResponse =
            serde_json::from_str(r#"{"success":true}"#).expect("tolerant deserialize");
        assert!(success.is_success());

        let failed: BrowserResponse =
            serde_json::from_str(r#"{"success":false}"#).expect("tolerant deserialize");
        assert!(!failed.is_success());

        // Failure precedence — an explicit false on either key loses over a
        // contradicting success key.
        let mixed: BrowserResponse =
            serde_json::from_str(r#"{"success":false,"ok":true}"#).expect("tolerant deserialize");
        assert!(!mixed.is_success());
    }

    #[test]
    fn verdict_covers_both_envelopes() {
        let v = |json: serde_json::Value| BrowserResponse::from_value(&json).verdict();
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
    fn browser_response_parses_structured_error_retryable() {
        let v = serde_json::json!({
            "success": false, "error": "boom", "code": "connection_failed", "retryable": true,
        });
        let resp = BrowserResponse::from_value(&v);
        assert_eq!(resp.success, Some(false));
        assert_eq!(resp.error.as_deref(), Some("boom"));
        assert_eq!(resp.code.as_deref(), Some("connection_failed"));
        assert_eq!(resp.retryable, Some(true));
        assert!(!resp.is_success());

        let v = serde_json::json!({
            "ok": false, "code": "connection_failed", "retryable": false,
        });
        let resp = BrowserResponse::from_value(&v);
        assert_eq!(resp.ok, Some(false));
        assert_eq!(resp.retryable, Some(false));
        assert!(!resp.is_success());
    }

    #[test]
    fn from_value_tolerates_unknown_keys_and_non_objects() {
        let resp = BrowserResponse::from_value(&serde_json::json!({
            "success": true, "version": 99, "unknown": [1, 2, 3],
        }));
        assert_eq!(resp.success, Some(true));
        assert!(resp.is_success());
        // Tolerates a payload with neither verdict key plus unknown keys.
        let resp = BrowserResponse::from_value(&serde_json::json!({
            "data": {"x": 1}, "future_key": true,
        }));
        assert_eq!(resp.verdict(), None);
        assert_eq!(resp.data, Some(serde_json::json!({"x": 1})));
        // A non-object / unparseable payload yields a default response.
        assert_eq!(
            BrowserResponse::from_value(&serde_json::json!("plain string")).verdict(),
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
}
