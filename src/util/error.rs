//! Structured error type preserving HTTP metadata from HTTP API calls.
//!
//! Wraps HTTP error details so downstream error classification can extract
//! typed fields (status code, response body, Retry-After header) instead of
//! relying solely on string-parsing of formatted error messages.
//!
//! Construct at the HTTP boundary — the point where a `reqwest::Response`
//! is available — so the typed data is captured before being lost to
//! stringification in the `anyhow::Result` boundary.

use std::fmt;
use std::time::Duration;

/// Structured error preserving HTTP metadata from API calls.
///
/// Once constructed, the [`HttpError::status`], [`HttpError::body`], and
/// [`HttpError::retry_after_ms`] fields are available to error classifiers
/// via `err.downcast_ref::<HttpError>()`.
///
/// Errors from third-party code or non-HTTP sources arrive without this
/// wrapper and are classified as retryable by default.
#[derive(Debug)]
pub struct HttpError {
    /// The HTTP status code (e.g., 429, 400, 500).
    pub status: u16,
    /// The response body text.
    pub body: String,
    /// Optional Retry-After duration in milliseconds (from the response header).
    pub retry_after_ms: Option<u64>,
    /// Provider or operation name for display purposes.
    pub context: String,
}

impl HttpError {
    /// Create a new [`HttpError`] with the given fields.
    #[must_use]
    #[expect(
        dead_code,
        reason = "Public constructor; kept for API completeness, not currently called"
    )]
    pub fn new(
        status: u16,
        context: impl Into<String>,
        body: impl Into<String>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self {
            status,
            context: context.into(),
            body: body.into(),
            retry_after_ms,
        }
    }

    /// Build a [`HttpError`] from a `reqwest::Response`, consuming it.
    ///
    /// Extracts HTTP status, response body, and the `Retry-After` header.
    pub async fn from_response(response: reqwest::Response, context: impl Into<String>) -> Self {
        let context: String = context.into();
        let status = response.status().as_u16();
        let retry_after_ms = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after_value);
        let body = response.text().await.unwrap_or_else(|e| {
            tracing::warn!(
                ?e,
                status,
                context,
                "Failed to read HTTP error response body"
            );
            String::new()
        });
        Self {
            status,
            body,
            retry_after_ms,
            context,
        }
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

/// Parse a `Retry-After` header value into milliseconds.
///
/// Accepts decimal seconds (e.g., `"5"`, `"2.5"`, `"120"`).
/// Rejects negative, NaN, or infinite values.
pub(crate) fn parse_retry_after_value(value: &str) -> Option<u64> {
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
    fn http_error_display_format() {
        let err = HttpError::new(429, "OpenAI", "Rate limit exceeded", None);
        let msg = err.to_string();
        assert!(msg.contains("OpenAI API error (429)"));
        assert!(msg.contains("Rate limit exceeded"));
    }

    #[test]
    fn http_error_downcast() {
        let err = anyhow::Error::from(HttpError::new(400, "test", "bad request body", None));
        assert_eq!(err.downcast_ref::<HttpError>().map(|e| e.status), Some(400),);

        // Non-structured error should not downcast
        let plain_err = anyhow::anyhow!("some other error");
        assert!(plain_err.downcast_ref::<HttpError>().is_none());
    }

    #[test]
    fn http_error_retry_after_extraction() {
        let err = HttpError::new(429, "test", "rate limited", Some(5000));
        assert_eq!(err.retry_after_ms, Some(5000));
        assert_eq!(err.status, 429);
    }

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
}
