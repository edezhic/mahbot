//! Provider error classification: one error becomes `ErrorClass::Retryable` or
//! `ErrorClass::NonRetryable`, and the hint tables below resolve the cases a
//! status code alone gets wrong. The split decides whether an attempt continues
//! or aborts immediately, so a doomed call does not burn the retry budget.
//!
//! Not a retry layer — there is no loop, no backoff and no attempt counter
//! here. The retry authority is `crate::retry`; this module's verdict reaches
//! it through `providers::failure_class`, which maps it onto `FailureClass`.

use crate::util::error::HttpError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorClass {
    /// A transient error that may resolve with retries (timeouts, 5xx, etc.).
    Retryable,
    /// A non-retryable client error (auth, invalid model, billing/quota exhausted,
    /// tool schema validation failure, etc.).
    NonRetryable,
}

/// Body-text hints that indicate a 429 is retryable even when its body also
/// reads as billing or quota exhaustion.
///
/// Consulted only for a typed [`HttpError`] carrying status 429 Too Many
/// Requests: a 4xx other than 408 and 429 is classified non-retryable before
/// any hint is consulted, so none here can override one. Every other status,
/// and any error that is not a typed [`HttpError`], is retryable without
/// consulting these hints at all.
///
/// These overrides handle cases where upstream proxy providers (e.g. OpenRouter)
/// forward error metadata from transient upstream rate limits that coincidentally
/// contain keywords like `insufficient_quota` in non-permanent contexts
/// (e.g. `metadata.provider_error_code` forwarded from Alibaba).
///
/// These hints are checked **before** [`NON_RETRYABLE_HINTS`], so a 429 body
/// that matches both is classified retryable.
const RETRYABLE_OVERRIDES: &[&str] = &[
    // Upstream provider reports a transient rate limit in its raw error
    // text (e.g. Alibaba/Qwen forwarded by OpenRouter).
    "temporarily rate-limited",
    // Upstream provider shared pool exhausted — transient, not account-level.
    "upstream_provider_shared_pool",
];

/// Body-text hints that indicate permanent (non-retryable) errors when the
/// HTTP status-code check is ambiguous — specifically, HTTP 429 Too Many
/// Requests is excluded from the status-based classification (rate limits
/// are transient), so these billing/quota hints override 429 to prevent
/// endless retries on exhausted accounts.
///
/// These hints are checked **after** [`RETRYABLE_OVERRIDES`], so any body
/// that matches both a retryable override and a non-retryable hint will be
/// correctly classified as retryable — preventing false positives when
/// upstream proxy providers embed billing error codes in transient rate-limit
/// metadata.
///
/// All other non-retryable errors (context window exceeded, tool schema
/// validation, auth failures) are reliably caught by the HTTP 4xx status-code
/// check (step 1 in [`classify_err`]) and do NOT need entries here.  That
/// also fixes a latent bug: 5xx responses whose body happens to contain a
/// hint-like substring are now correctly classified as retryable.
const NON_RETRYABLE_HINTS: &[&str] = &[
    "insufficient balance",
    "insufficient_quota",
    "quota exhausted",
    "quota exceeded",
    "error code 1113",
];

/// Classify an error into one of the [`ErrorClass`] variants.
///
/// The classification cascade is:
/// 1. **4xx status codes** (except 408 Request Timeout and 429 Too Many Requests)
///    — structured [`HttpError`] downcast.  Clear client errors that never
///    self-heal.
/// 2. **HTTP 429 — body-text disambiguation**: first checks
///    [`RETRYABLE_OVERRIDES`] (upstream transient rate-limit signals take
///    priority), then [`NON_RETRYABLE_HINTS`] (billing/quota exhaustion).
/// 3. Default to [`Retryable`](ErrorClass::Retryable).
pub(crate) fn classify_err(err: &anyhow::Error) -> ErrorClass {
    // ── Typed path: use structured fields from HttpError directly ──
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        let body_lower = http_err.body.to_lowercase();

        // ── Step 1: 4xx status codes (except 408 Request Timeout and
        //    429 Too Many Requests) ──
        // These are clear client errors (auth, invalid model, context
        // window, tool schema validation, etc.) and are never retryable.
        if (400..500).contains(&http_err.status) && http_err.status != 408 && http_err.status != 429
        {
            return ErrorClass::NonRetryable;
        }

        // ── Step 2: HTTP 429 — ambiguous; delegate to body-text hints ──
        if http_err.status == 429 {
            // 2a. Retryable overrides take priority. Upstream providers
            //     sometimes embed rate-limit info in metadata that happens
            //     to contain billing keywords (e.g. `insufficient_quota` in
            //     OpenRouter's `metadata.provider_error_code`).  Override
            //     signals (e.g. "temporarily rate-limited") indicate the
            //     failure is a transient upstream rate limit, not account
            //     exhaustion.
            if RETRYABLE_OVERRIDES.iter().any(|h| body_lower.contains(h)) {
                return ErrorClass::Retryable;
            }
            // 2b. Billing/quota body-text hints — permanent account
            //     exhaustion, not transient rate limiting.
            if NON_RETRYABLE_HINTS.iter().any(|h| body_lower.contains(h)) {
                return ErrorClass::NonRetryable;
            }
        }

        return ErrorClass::Retryable;
    }
    ErrorClass::Retryable
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an [`HttpError`] with context="test", reducing boilerplate in
    /// error-classification tests.
    fn test_err(status: u16, body: &str) -> anyhow::Error {
        anyhow::Error::from(HttpError {
            status,
            body: body.into(),
            context: "test".into(),
        })
    }

    // ── Error classification unit tests ───────────────────────

    #[test]
    fn retryable_error_classification() {
        let is_non_retryable =
            |e: &anyhow::Error| matches!(classify_err(e), ErrorClass::NonRetryable);
        // Non-retryable via status code (HttpError 4xx, excluding 408/429)
        assert!(is_non_retryable(&test_err(401, "Unauthorized")));
        assert!(is_non_retryable(&test_err(403, "Forbidden")));
        assert!(is_non_retryable(&test_err(400, "invalid api key")));
        // Non-retryable via billing/quota hints (override 429)
        assert!(is_non_retryable(&test_err(429, "insufficient balance")));
        assert!(is_non_retryable(&test_err(429, "insufficient_quota")));
        assert!(is_non_retryable(&test_err(429, "quota exhausted")));
        assert!(is_non_retryable(&test_err(429, "error code 1113")));
        // Retryable — no HttpError, no hint match
        assert!(!is_non_retryable(&anyhow::anyhow!("500 Server Error")));
        assert!(!is_non_retryable(&anyhow::anyhow!("502 Bad Gateway")));
        assert!(!is_non_retryable(&anyhow::anyhow!(
            "503 Service Unavailable"
        )));
        assert!(!is_non_retryable(&anyhow::anyhow!("connection reset")));
        assert!(!is_non_retryable(&anyhow::anyhow!(
            "model overloaded, try again later"
        )));
    }

    #[test]
    fn classify_err_typed_path() {
        // ── HttpError typed path for classify_err ──

        // 429 transient rate limit → retryable (no billing/quota hint match,
        // so falls through to return ErrorClass::Retryable)
        assert!(matches!(
            classify_err(&test_err(429, "Too Many Requests")),
            ErrorClass::Retryable
        ));
        assert!(matches!(
            classify_err(&test_err(429, "rate limit exceeded")),
            ErrorClass::Retryable
        ));

        // 408 Request Timeout → retryable (excluded from 4xx status check)
        assert!(matches!(
            classify_err(&test_err(408, "Request Timeout")),
            ErrorClass::Retryable
        ));

        // OpenRouter 502 "invalid response" → NOT NonRetryable
        // (the word "invalid" alone does not imply a bad model id)
        assert_eq!(
            classify_err(&test_err(
                502,
                "Your chosen model is down or we received an invalid response from it"
            )),
            ErrorClass::Retryable
        );

        // Regression: 5xx HttpError with "model not found" in body → Retryable
        // (the typed path returns Retryable for non-4xx, non-429 responses
        // regardless of body content)
        assert_eq!(
            classify_err(&test_err(502, "upstream model not found")),
            ErrorClass::Retryable
        );
    }

    // ── Tool schema error detection tests ───────────────────────────────

    #[test]
    fn non_retryable_hints_are_classified_non_retryable() {
        for hint in NON_RETRYABLE_HINTS {
            let err = test_err(429, hint);
            assert!(
                matches!(classify_err(&err), ErrorClass::NonRetryable),
                "hint '{hint}' should be classified as NonRetryable"
            );
        }
    }

    #[test]
    fn proxy_5xx_with_hint_text_is_retryable() {
        // Regression: 5xx responses from proxy providers (e.g. OpenRouter
        // forwarding an upstream error) may contain billing/quota language
        // in the body. These must remain Retryable — the issue is a
        // transient upstream failure, not account exhaustion.
        for hint in NON_RETRYABLE_HINTS {
            let err = test_err(502, &format!("upstream error: {hint}"));
            assert!(
                matches!(classify_err(&err), ErrorClass::Retryable),
                "502 with hint '{hint}' should remain Retryable"
            );
        }
    }

    /// A 429 that mentions account exhaustion is still the upstream provider's
    /// rate limit when the envelope also carries the provider's own retry signal:
    /// the retry override must outrank the billing hint, or the agent aborts on a
    /// limit that would have cleared by the next attempt.
    #[test]
    fn an_upstream_retryable_override_beats_the_quota_hint() {
        // (label, provider, the provider's own retry signal)
        let cases = [
            (
                "rate-limited upstream",
                "Qwen",
                r#""raw":"upstream error: temporarily rate-limited upstream, please retry""#,
            ),
            (
                "shared pool",
                "Alibaba",
                r#""limit_source":"upstream_provider_shared_pool""#,
            ),
        ];
        for (label, provider, signal) in cases {
            let body = format!(
                concat!(
                    r#"{{"error":{{"code":429,"message":"Provider returned error","metadata":"#,
                    r#"{{"provider_name":"{}","provider_error_code":"insufficient_quota","#,
                    r#"{}}}}}}}"#,
                ),
                provider, signal
            );
            let err = test_err(429, &body);
            assert!(
                matches!(classify_err(&err), ErrorClass::Retryable),
                "{label}: 429 with {signal} should be Retryable despite 'insufficient_quota'"
            );
        }
    }

    #[test]
    fn genuine_quota_exhaustion_still_non_retryable() {
        // Verify that a genuine quota-exhaustion 429 (no retryable override
        // signal) remains NonRetryable — the fix must not break existing
        // billing detection.
        let body = r#"{"error":{"code":429,"message":"You have exceeded your quota. insufficient_quota"}}"#;
        let err = test_err(429, body);
        assert!(
            matches!(classify_err(&err), ErrorClass::NonRetryable),
            "genuine quota exhaustion 429 without override should remain NonRetryable"
        );
    }

    #[test]
    fn retryable_overrides_do_not_affect_non_429() {
        // Retryable override signals in non-429/5xx contexts must not change
        // classification — e.g., a 400 with "temporarily rate-limited" in body
        // is still a client error (NonRetryable via status code).
        let err = test_err(400, "Bad Request: temporarily rate-limited");
        assert!(
            matches!(classify_err(&err), ErrorClass::NonRetryable),
            "400 with override text should still be NonRetryable"
        );
    }
}
