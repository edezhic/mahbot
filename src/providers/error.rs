//! Structured error type preserving HTTP metadata from provider API calls.
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

/// Structured error preserving HTTP metadata from provider API calls.
///
/// Once constructed, the [`ProviderError::status`], [`ProviderError::body`], and [`ProviderError::retry_after_ms`] fields
/// are available to `classify_err`
/// via `err.downcast_ref::<ProviderError>()`.
///
/// Errors from third-party code or non-HTTP sources arrive without this
/// wrapper and fall back to the existing string-parsing path unchanged.
#[derive(Debug)]
pub struct ProviderError {
    /// The HTTP status code (e.g., 429, 400, 500).
    pub status: u16,
    /// The response body text.
    pub body: String,
    /// Optional Retry-After duration in milliseconds (from the response header).
    pub retry_after_ms: Option<u64>,
    /// Provider name for display purposes.
    provider: String,
}

impl ProviderError {
    /// Create a new [`ProviderError`] with the given fields.
    #[must_use]
    pub fn new(
        status: u16,
        provider: impl Into<String>,
        body: impl Into<String>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self {
            status,
            provider: provider.into(),
            body: body.into(),
            retry_after_ms,
        }
    }

    /// Build a [`ProviderError`] from a `reqwest::Response`, consuming it.
    ///
    /// Extracts HTTP status, response body, and the `Retry-After` header.
    pub async fn from_response(response: reqwest::Response, provider: impl Into<String>) -> Self {
        let status = response.status().as_u16();
        let retry_after_ms = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after_value);
        let body = response.text().await.unwrap_or_default();
        Self {
            status,
            provider: provider.into(),
            body,
            retry_after_ms,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} API error ({}): {}",
            self.provider, self.status, self.body
        )
    }
}

impl std::error::Error for ProviderError {}

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
    fn provider_error_display_format() {
        let err = ProviderError::new(429, "OpenAI", "Rate limit exceeded", None);
        let msg = err.to_string();
        assert!(msg.contains("OpenAI API error (429)"));
        assert!(msg.contains("Rate limit exceeded"));
    }

    #[test]
    fn provider_error_downcast() {
        let err = anyhow::Error::from(ProviderError::new(400, "test", "bad request body", None));
        let downcasted = err.downcast_ref::<ProviderError>();
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().status, 400);

        // Non-structured error should not downcast
        let plain_err = anyhow::anyhow!("some other error");
        assert!(plain_err.downcast_ref::<ProviderError>().is_none());
    }

    #[test]
    fn provider_error_retry_after_extraction() {
        let err = ProviderError::new(429, "test", "rate limited", Some(5000));
        assert_eq!(err.retry_after_ms, Some(5000));

        let err = ProviderError::new(200, "test", "ok", None);
        assert_eq!(err.retry_after_ms, None);
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
