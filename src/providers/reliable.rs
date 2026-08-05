use crate::util::error::HttpError;

// ── Error Classification ─────────────────────────────────────────────────
// Errors are split into retryable (transient server/network failures) and
// non-retryable (permanent client errors). This distinction drives whether
// the retry loop continues or aborts immediately — avoiding wasted latency
// on errors that cannot self-heal.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorClass {
    /// A transient error that may resolve with retries (timeouts, 5xx, etc.).
    Retryable,
    /// A non-retryable client error (auth, invalid model, billing/quota exhausted,
    /// tool schema validation failure, etc.).
    NonRetryable,
}

/// Body-text hints that indicate the error is retryable even when other
/// signals (like `NON_RETRYABLE_HINTS` or 4xx status codes) suggest otherwise.
///
/// These overrides handle cases where upstream proxy providers (e.g. OpenRouter)
/// forward error metadata from transient upstream rate limits that coincidentally
/// contain keywords like `insufficient_quota` in non-permanent contexts
/// (e.g. `metadata.provider_error_code` forwarded from Alibaba).
///
/// Priority: retryable overrides are checked first, then non-retryable hints,
/// then status-code defaults. If ANY retryable override matches, the error
/// is classified as Retryable regardless of other indicators.
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

    // ── Context window error handling ─────────────────────────

    #[test]
    fn context_window_error_classification() {
        let is_non_retryable =
            |e: &anyhow::Error| matches!(classify_err(e), ErrorClass::NonRetryable);
        // Context window exceeded — NonRetryable via status 400
        assert!(is_non_retryable(&test_err(
            400,
            "request (8968 tokens) exceeds the available context size (8448 tokens)",
        )));
        assert!(is_non_retryable(&test_err(
            400,
            "This model's maximum context length is 8192 tokens",
        )));
        assert!(is_non_retryable(&test_err(
            400,
            "maximum context length of this model is 128K tokens",
        )));
        // 4xx errors are still non-retryable via status code
        assert!(is_non_retryable(&test_err(401, "Unauthorized")));
    }

    // ── Tool schema error detection tests ───────────────────────────────

    #[test]
    fn tool_schema_error_detection() {
        use ErrorClass::NonRetryable;
        // Detects various tool schema error patterns as NonRetryable via status 400
        for msg in [
            r#"Groq API error (400 Bad Request): {"error":{"message":"tool call validation failed: attempted to call tool 'recall' which was not in request"}}"#,
            "tool 'search' which was not in request",
            "function 'foo' not found in tool list",
            "invalid_tool_call: no matching function",
        ] {
            assert!(
                matches!(classify_err(&test_err(400, msg)), NonRetryable),
                "should detect: {msg}"
            );
        }
        // Pure 400 without tool-schema keywords → also NonRetryable (via status code)
        assert!(
            matches!(
                classify_err(&test_err(400, "invalid api key provided")),
                NonRetryable
            ),
            "pure 400 should be NonRetryable"
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

    #[test]
    fn upstream_rate_limit_overrides_quota_hint() {
        // Regression: OpenRouter forwards upstream provider rate limits with
        // "insufficient_quota" in metadata.provider_error_code but "temporarily
        // rate-limited" in the raw upstream error text.  The retryable override
        // must take priority over the non-retryable billing hint so the agent
        // retries instead of aborting.
        let body = concat!(
            r#"{"error":{"code":429,"message":"Provider returned error","metadata":"#,
            r#"{"provider_name":"Qwen","provider_error_code":"insufficient_quota","#,
            r#""raw":"upstream error: temporarily rate-limited upstream, please retry"}}}"#,
        );
        let err = test_err(429, body);
        assert!(
            matches!(classify_err(&err), ErrorClass::Retryable),
            "429 with 'temporarily rate-limited' in metadata.raw should be Retryable \
             despite also containing 'insufficient_quota'"
        );
    }

    #[test]
    fn upstream_shared_pool_overrides_quota_hint() {
        // Similar to the above but with "upstream_provider_shared_pool" as the
        // retryable override signal (another variant of upstream exhaustion).
        let body = concat!(
            r#"{"error":{"code":429,"message":"Provider returned error","metadata":"#,
            r#"{"provider_name":"Alibaba","provider_error_code":"insufficient_quota","#,
            r#""limit_source":"upstream_provider_shared_pool"}}}"#,
        );
        let err = test_err(429, body);
        assert!(
            matches!(classify_err(&err), ErrorClass::Retryable),
            "429 with 'upstream_provider_shared_pool' should be Retryable \
             despite also containing 'insufficient_quota'"
        );
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
