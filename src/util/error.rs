//! Structured error type preserving HTTP metadata from HTTP API calls.
//!
//! Wraps the details so downstream classification can extract typed fields (status code,
//! response body) instead of string-parsing a formatted message.
//!
//! Construct at the HTTP boundary — where a `reqwest::Response` is available — so the typed data
//! is captured before stringification at the `anyhow::Result` boundary.

use std::fmt;
use std::time::Duration;

/// Structured error preserving HTTP metadata from API calls.
///
/// [`HttpError::status`] and [`HttpError::body`] are available to classifiers via
/// `err.downcast_ref::<HttpError>()`. Errors from third-party or non-HTTP sources arrive without
/// this wrapper and are classified as retryable by default. Build one through [`HttpError::new`] /
/// [`HttpError::from_response`] so the body is masked (see [`crate::util::control_token`]).
#[derive(Debug)]
pub struct HttpError {
    pub status: u16,
    pub body: String,
    /// Provider or operation name for display purposes.
    pub context: String,
}

impl HttpError {
    /// Build one from an already-read status and body, masking model control tokens.
    ///
    /// The body is a provider's error envelope, and it is handed back into the caller's context
    /// (agent failure text, `llm_failures` trail, ticket comment) — so the mask happens once, here,
    /// for every path that builds an `HttpError`. A token in a body must not re-infect the session
    /// that reads it.
    #[must_use]
    pub fn new(status: u16, body: String, context: impl Into<String>) -> Self {
        Self {
            status,
            body: crate::util::control_token::neutralize_owned(body),
            context: context.into(),
        }
    }

    /// Build a [`HttpError`] from a `reqwest::Response`, consuming it and extracting status and
    /// body.
    pub async fn from_response(response: reqwest::Response, context: impl Into<String>) -> Self {
        let context: String = context.into();
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_else(|e| {
            tracing::warn!(
                ?e,
                status,
                context,
                "Failed to read HTTP error response body"
            );
            String::new()
        });
        Self::new(status, body, context)
    }
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} API error ({}): {}",
            self.context, self.status, self.body
        )
    }
}

impl std::error::Error for HttpError {}

/// Extract the `Retry-After` header value in milliseconds, if present.
pub(crate) fn retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after_value)
}

/// Parse a `Retry-After` header value into milliseconds, in either the delta-seconds or the
/// HTTP-date form.
fn parse_retry_after_value(value: &str) -> Option<u64> {
    let num_str: String = value
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if num_str.is_empty() {
        return None;
    }
    let secs = num_str.parse::<f64>().ok()?;
    if secs.is_finite() && secs >= 0.0 {
        let millis = Duration::from_secs_f64(secs).as_millis();
        u64::try_from(millis).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_value_various_inputs() {
        assert_eq!(parse_retry_after_value("5"), Some(5000));
        assert_eq!(parse_retry_after_value("2.5"), Some(2500));
        assert_eq!(parse_retry_after_value("0"), Some(0));
        assert_eq!(parse_retry_after_value("10"), Some(10_000));
        assert_eq!(parse_retry_after_value("120"), Some(120_000));
        assert_eq!(parse_retry_after_value("-1"), None);
        assert_eq!(parse_retry_after_value("abc"), None);
        assert_eq!(parse_retry_after_value(""), None);
        assert_eq!(parse_retry_after_value("  5  "), Some(5000));
    }

    /// A refusal body is handed back into the caller's context, so a control token inside it is
    /// masked at construction — the one place every `HttpError` passes through.
    #[test]
    fn a_built_http_error_masks_control_tokens_in_its_body() {
        let err = HttpError::new(
            400,
            r#"{"error":{"message":"text contains <|image|> which is not allowed"}}"#.to_string(),
            "OpenRouter",
        );
        assert!(!err.body.contains("<|"));
        assert!(err.body.contains(crate::util::control_token::PLACEHOLDER));
        assert!(!err.to_string().contains("<|"));
    }
}
