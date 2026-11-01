//! Provider error classification: one error becomes `ErrorClass::Retryable` or
//! `ErrorClass::NonRetryable`, and the hint tables below resolve the cases a status code alone gets
//! wrong. The split decides whether an attempt continues or aborts immediately, so a doomed call
//! does not burn the retry budget. Not a retry layer — no loop, no backoff, no attempt counter: the
//! retry authority is `crate::retry`, and this module's verdict reaches it through
//! `providers::failure_class`.

use crate::util::error::HttpError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorClass {
    /// A transient error that may resolve with retries (timeouts, 5xx, …).
    Retryable,
    /// A non-retryable client error (auth, invalid model, billing/quota exhausted, tool schema
    /// validation).
    NonRetryable,
}

/// Body-text hints that make a 429 retryable even when its body also reads as billing or quota
/// exhaustion. Consulted only for a typed [`HttpError`] carrying 429: a 4xx other than 408 and 429
/// is non-retryable before any hint is consulted, and every other status — or any error that is not
/// a typed [`HttpError`] — is retryable without consulting them at all. They cover upstream proxy
/// providers (e.g. OpenRouter) forwarding metadata from transient upstream limits that
/// coincidentally contains keywords like `insufficient_quota` (e.g. `metadata.provider_error_code`
/// from Alibaba). Checked **before** [`NON_RETRYABLE_HINTS`], so a 429 body matching both is
/// retryable.
const RETRYABLE_OVERRIDES: &[&str] = &[
    // Upstream provider reports a transient rate limit (e.g. Alibaba/Qwen forwarded by OpenRouter).
    "temporarily rate-limited",
    // Upstream provider shared pool exhausted — transient, not account-level.
    "upstream_provider_shared_pool",
];

/// Body-text hints that indicate permanent (non-retryable) errors when the status code is ambiguous
/// — specifically, 429 is excluded from the status check (rate limits are transient), so these
/// billing/quota hints override it to prevent endless retries on exhausted accounts. Checked
/// **after** [`RETRYABLE_OVERRIDES`], so a body matching both is retryable — no false positives
/// when a proxy embeds billing codes in transient rate-limit metadata. Every other non-retryable
/// error (context window, tool schema, auth) is caught by the 4xx check and needs no entry, which
/// also keeps a 5xx whose body contains a hint-like substring retryable.
const NON_RETRYABLE_HINTS: &[&str] = &[
    "insufficient balance",
    "insufficient_quota",
    "quota exhausted",
    "quota exceeded",
    "error code 1113",
];

/// Classify an error into one of the [`ErrorClass`] variants, in this cascade: (1) 4xx status codes
/// other than 408 and 429 — clear client errors that never self-heal; (2) 429, disambiguated by
/// body text — [`RETRYABLE_OVERRIDES`] first (transient upstream rate limits), then
/// [`NON_RETRYABLE_HINTS`] (billing/quota exhaustion); (3) otherwise retryable.
pub(crate) fn classify_err(err: &anyhow::Error) -> ErrorClass {
    // Typed path: use the structured fields from HttpError directly.
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        let body_lower = http_err.body.to_lowercase();

        // Step 1: 4xx other than 408 and 429 — auth, invalid model, context window, tool schema:
        // never retryable.
        if (400..500).contains(&http_err.status) && http_err.status != 408 && http_err.status != 429
        {
            return ErrorClass::NonRetryable;
        }

        // Step 2: 429 is ambiguous — delegate to the body-text hints.
        if http_err.status == 429 {
            // 2a. Retryable overrides take priority: a proxy may embed billing keywords in
            // transient metadata (e.g. `insufficient_quota` in OpenRouter's
            // `metadata.provider_error_code`), while a "temporarily rate-limited" signal means the
            // failure is a transient upstream limit, not account exhaustion.
            if RETRYABLE_OVERRIDES.iter().any(|h| body_lower.contains(h)) {
                return ErrorClass::Retryable;
            }
            // 2b. Billing/quota body text — permanent account exhaustion, not transient rate
            // limiting.
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

    /// Builds an [`HttpError`] with context="test", reducing boilerplate in the classification
    /// tests.
    fn test_err(status: u16, body: &str) -> anyhow::Error {
        anyhow::Error::from(HttpError {
            status,
            body: body.into(),
            context: "test".into(),
        })
    }

    #[test]
    fn retryable_error_classification() {
        let is_non_retryable =
            |e: &anyhow::Error| matches!(classify_err(e), ErrorClass::NonRetryable);
        // Non-retryable via status code (HttpError 4xx, excluding 408/429).
        assert!(is_non_retryable(&test_err(401, "Unauthorized")));
        assert!(is_non_retryable(&test_err(403, "Forbidden")));
        assert!(is_non_retryable(&test_err(400, "invalid api key")));
        // Non-retryable via billing/quota hints (which override 429).
        assert!(is_non_retryable(&test_err(429, "insufficient balance")));
        assert!(is_non_retryable(&test_err(429, "insufficient_quota")));
        assert!(is_non_retryable(&test_err(429, "quota exhausted")));
        assert!(is_non_retryable(&test_err(429, "error code 1113")));
        // Retryable — no HttpError, no hint match.
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
        // 429 transient rate limit → retryable (no billing/quota hint match, so it falls through).

        assert!(matches!(
            classify_err(&test_err(429, "Too Many Requests")),
            ErrorClass::Retryable
        ));
        assert!(matches!(
            classify_err(&test_err(429, "rate limit exceeded")),
            ErrorClass::Retryable
        ));

        assert!(matches!(
            classify_err(&test_err(408, "Request Timeout")),
            ErrorClass::Retryable
        ));

        // OpenRouter 502 "invalid response" → NOT NonRetryable: the word "invalid" alone does not
        // imply a bad model id.
        assert_eq!(
            classify_err(&test_err(
                502,
                "Your chosen model is down or we received an invalid response from it"
            )),
            ErrorClass::Retryable
        );

        // Regression: a 5xx HttpError with "model not found" in the body stays Retryable — the
        // typed path returns Retryable for non-4xx, non-429 responses whatever the body says.
        assert_eq!(
            classify_err(&test_err(502, "upstream model not found")),
            ErrorClass::Retryable
        );
    }

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
        // Regression: 5xx responses from proxy providers may contain billing/quota language in the
        // body. These must stay Retryable — a transient upstream failure, not account exhaustion.
        for hint in NON_RETRYABLE_HINTS {
            let err = test_err(502, &format!("upstream error: {hint}"));
            assert!(
                matches!(classify_err(&err), ErrorClass::Retryable),
                "502 with hint '{hint}' should remain Retryable"
            );
        }
    }

    /// A 429 mentioning account exhaustion is still the upstream provider's rate limit when the
    /// envelope also carries the provider's own retry signal: the override must outrank the billing
    /// hint, or the agent aborts on a limit that would have cleared by the next attempt.
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
        // A genuine quota-exhaustion 429 (no retryable override signal) stays NonRetryable — the
        // fix must not break existing billing detection.
        let body = r#"{"error":{"code":429,"message":"You have exceeded your quota. insufficient_quota"}}"#;
        let err = test_err(429, body);
        assert!(
            matches!(classify_err(&err), ErrorClass::NonRetryable),
            "genuine quota exhaustion 429 without override should remain NonRetryable"
        );
    }

    #[test]
    fn retryable_overrides_do_not_affect_non_429() {
        // Retryable override signals outside the 429 context change nothing — a 400 with
        // "temporarily rate-limited" is still a client error (NonRetryable by status code).
        let err = test_err(400, "Bad Request: temporarily rate-limited");
        assert!(
            matches!(classify_err(&err), ErrorClass::NonRetryable),
            "400 with override text should still be NonRetryable"
        );
    }
}
