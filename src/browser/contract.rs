//! One place for the chrome-use response envelope parsing, shared by the
//! interactive `browser` tool and the `mahbot browser` CLI.
//!
//! chrome-use answers `--json` commands with either the classic `success` key
//! or the newer `ok` key. Every consumer needs the same conservative verdict
//! logic (failure-precedence), so the typed [`BrowserResponse`] and the
//! `Value`-based [`envelope_verdict`] both delegate to [`envelope_success`] and
//! cannot drift. The `mahbot browser` stdout output contract ([`SCHEMA_VERSION`],
//! [`OutKind`], [`OutEnvelope`]) lives here too.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response from chrome-use `--json` commands.
#[derive(Debug, Deserialize)]
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
}

impl BrowserResponse {
    /// Success gate accepting both the classic `success: true` envelope and
    /// the latest `ok: true` one — delegates to the shared [`envelope_success`]
    /// predicate (failure precedence) so the typed and Value-based paths
    /// cannot drift.
    pub(crate) fn is_success(&self) -> bool {
        envelope_success(self.success, self.ok)
    }
}

/// Extract the `error` message from a CLI error response, if any.
pub(crate) fn extract_error(stdout: &[u8]) -> Option<String> {
    let v: Value = serde_json::from_slice(stdout).unwrap_or_default();
    v.get("error")
        .and_then(Value::as_str)
        .map(String::from)
        .filter(|s| !s.is_empty())
}

/// Core envelope-success predicate with failure precedence: an explicit
/// `false` on either key loses over a contradicting success key (conservative
/// — the error text surfaces), and a payload with neither key is not a
/// success. Shared by [`envelope_verdict`] (Value-based) and
/// `BrowserResponse::is_success` (typed) so the two cannot drift.
fn envelope_success(success: Option<bool>, ok: Option<bool>) -> bool {
    !(success == Some(false) || ok == Some(false)) && (success == Some(true) || ok == Some(true))
}

/// Tri-state verdict of a chrome-use JSON envelope: `Some(true)` when either
/// `success` or `ok` reports success, `Some(false)` when either reports
/// failure, `None` when the payload carries no verdict key (callers decide
/// their own default). Newer chrome-use commands replaced `success` with `ok`
/// (e.g. `session list`).
pub(crate) fn envelope_verdict(v: &Value) -> Option<bool> {
    let success = v.get("success").and_then(Value::as_bool);
    let ok = v.get("ok").and_then(Value::as_bool);
    if success.is_none() && ok.is_none() {
        return None;
    }
    Some(envelope_success(success, ok))
}

/// Extract textual content from an chrome-use snapshot response `data` field.
///
/// chrome-use can return the snapshot as:
/// - A plain string (via `snapshot -c`)
/// - An object with a `content` field (via `get_text`)
/// - An object with `origin`, `refs`, and `snapshot` fields (via `open` auto-snapshot)
///
/// Returns `None` if none of these shapes match.
pub(crate) fn extract_snapshot_text(data: &serde_json::Value) -> Option<String> {
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

        // Failure precedence — mirrors envelope_verdict: an explicit false on
        // either key loses over a contradicting success key.
        let mixed: BrowserResponse =
            serde_json::from_str(r#"{"success":false,"ok":true}"#).expect("tolerant deserialize");
        assert!(!mixed.is_success());
    }

    #[test]
    fn envelope_verdict_covers_both_envelopes() {
        let v = |json: serde_json::Value| envelope_verdict(&json);
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
