//! Provider error classification: one error becomes one [`ErrorClass`], and the hint tables below
//! resolve the cases a status code alone gets wrong. The split decides whether an attempt continues
//! or aborts immediately, so a doomed call does not burn the retry budget. Not a retry layer — no
//! loop, no backoff, no attempt counter: the retry authority is `crate::retry`, and this module's
//! verdict reaches it through `providers::failure_class`.

use crate::util::error::HttpError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorClass {
    /// A transient error that may resolve with retries (timeouts, 5xx, 429, …).
    Retryable,
    /// The request itself is unacceptable (auth, invalid model, billing/quota exhausted, tool
    /// schema, context length).
    InvalidRequest,
    /// The provider refused what the request *contained* — content inspection, a leaked control
    /// token, a content-policy violation. Terminal: the same text is refused again.
    ContentRejection,
    /// The provider's guardrail/safety layer refused to answer. Terminal.
    Guardrail,
}

/// Body-text hints that make a 429 retryable even when its body also reads as billing or quota
/// exhaustion. Consulted only for a typed [`HttpError`] carrying 429 — a 4xx other than 408 and 429
/// is already terminal, and every other status (or any error that is not a typed [`HttpError`]) is
/// retryable without consulting them at all. They cover upstream proxy providers (e.g. OpenRouter)
/// forwarding metadata from transient upstream limits that coincidentally contains keywords like
/// `insufficient_quota` (e.g. `metadata.provider_error_code` from Alibaba). Checked **before**
/// [`NON_RETRYABLE_HINTS`], so a 429 body matching both is retryable.
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

/// Body-text vocabulary of a provider refusing the request's **content**. A leaked model control
/// token is the case this exists for: the provider names it in the rejection and refuses every
/// later request carrying it, so the class must be terminal and must name the cause — the history
/// repair keys on it.
const CONTENT_REJECTION_HINTS: &[&str] = &[
    "special token",
    "data_inspection_failed",
    "inappropriate content",
    "content_policy_violation",
    "invalid_image",
    "content policy",
];

/// Body-text vocabulary of a provider's safety layer refusing to answer. Terminal, like a content
/// rejection; the separate label keeps the two distinguishable in the failure trail. Deliberately
/// no bare "refusal": a proxy's "upstream refusal" is a transient failure to serve, not a verdict on
/// the request.
const GUARDRAIL_HINTS: &[&str] = &["guardrail", "moderation", "content_filter"];

/// The refusal class a body's wording names, if any. Consulted only where the status itself leaves
/// the class open (see [`classify_err`]): a 5xx whose body happens to mention moderation is an
/// upstream outage, and a 408 or 429 is a timeout or a rate limit whatever its body says.
fn refusal_class(body_lower: &str) -> Option<ErrorClass> {
    if CONTENT_REJECTION_HINTS
        .iter()
        .any(|h| body_lower.contains(h))
    {
        return Some(ErrorClass::ContentRejection);
    }
    GUARDRAIL_HINTS
        .iter()
        .any(|h| body_lower.contains(h))
        .then_some(ErrorClass::Guardrail)
}

/// The status a provider error envelope implies when it arrives inside a 2xx body and carries no
/// numeric code of its own: a content rejection is a request the provider will refuse again (400),
/// anything else is a failure while producing the output (502, retryable).
///
/// Only the content class is inferred from wording. A code-less body that merely mentions a
/// guardrail or a moderation service is not evidence that the request was refused — a guardrail
/// refusal arrives with a 4xx (or an `error.code`), and that status is honoured at
/// [`classify_err`]'s wording step.
#[must_use]
pub(crate) fn implied_status(body: &str) -> u16 {
    match refusal_class(&body.to_lowercase()) {
        Some(ErrorClass::ContentRejection) => 400,
        _ => 502,
    }
}

/// Classify an error into one of the [`ErrorClass`] variants, in this cascade: (1) the refusal
/// wording of [`refusal_class`] on a status below 500 that is neither 408 nor 429 — a 2xx body can
/// carry one (that is how OpenRouter reports a post-header rejection), and no retry clears it;
/// (2) any other 4xx — clear client errors that never self-heal; (3) 429, disambiguated by body
/// text — [`RETRYABLE_OVERRIDES`] first (transient upstream rate limits), then
/// [`NON_RETRYABLE_HINTS`] (billing/quota exhaustion); (4) otherwise retryable, 408 and every 5xx
/// included.
pub(crate) fn classify_err(err: &anyhow::Error) -> ErrorClass {
    // Typed path: use the structured fields from HttpError directly.
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        let body_lower = http_err.body.to_lowercase();

        // Step 1: a refusal. Checked before the status rules because a provider reports the same
        // rejection with a 400 or inside a 200 body (OpenRouter sends the post-header ones with
        // 200); the status is only a carrier, the vocabulary is the signal. A 408 and a 429 are the
        // two transient 4xx — their bodies routinely quote upstream metadata that can contain any of
        // this wording, so the status decides and they stay retryable.
        if http_err.status < 500
            && http_err.status != 408
            && http_err.status != 429
            && let Some(class) = refusal_class(&body_lower)
        {
            return class;
        }

        // Step 2: 4xx other than 408 and 429 — auth, invalid model, context window, tool schema:
        // never retryable.
        if (400..500).contains(&http_err.status) && http_err.status != 408 && http_err.status != 429
        {
            return ErrorClass::InvalidRequest;
        }

        // Step 3: 429 is ambiguous — delegate to the body-text hints.
        if http_err.status == 429 {
            // 3a. Retryable overrides take priority: a proxy may embed billing keywords in
            // transient metadata (e.g. `insufficient_quota` in OpenRouter's
            // `metadata.provider_error_code`), while a "temporarily rate-limited" signal means the
            // failure is a transient upstream limit, not account exhaustion.
            if RETRYABLE_OVERRIDES.iter().any(|h| body_lower.contains(h)) {
                return ErrorClass::Retryable;
            }
            // 3b. Billing/quota body text — permanent account exhaustion, not transient rate
            // limiting.
            if NON_RETRYABLE_HINTS.iter().any(|h| body_lower.contains(h)) {
                return ErrorClass::InvalidRequest;
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
        let is_terminal = |e: &anyhow::Error| classify_err(e) != ErrorClass::Retryable;
        // Terminal via status code (HttpError 4xx, excluding 408/429).
        assert!(is_terminal(&test_err(401, "Unauthorized")));
        assert!(is_terminal(&test_err(403, "Forbidden")));
        assert!(is_terminal(&test_err(400, "invalid api key")));
        // Terminal via billing/quota hints (which override 429).
        assert!(is_terminal(&test_err(429, "insufficient balance")));
        assert!(is_terminal(&test_err(429, "insufficient_quota")));
        assert!(is_terminal(&test_err(429, "quota exhausted")));
        assert!(is_terminal(&test_err(429, "error code 1113")));
        // Retryable — no HttpError, no hint match.
        assert!(!is_terminal(&anyhow::anyhow!("500 Server Error")));
        assert!(!is_terminal(&anyhow::anyhow!("502 Bad Gateway")));
        assert!(!is_terminal(&anyhow::anyhow!("503 Service Unavailable")));
        assert!(!is_terminal(&anyhow::anyhow!("connection reset")));
        assert!(!is_terminal(&anyhow::anyhow!(
            "model overloaded, try again later"
        )));
    }

    /// The terminal split: a refusal names its cause (content inspection or guardrail), a plain
    /// client error does not. A 2xx status does not exempt a refusal — that is exactly how the
    /// provider reports a post-header rejection, and retrying it is what burned the budget.
    #[test]
    fn refusals_are_split_into_content_and_guardrail_classes() {
        let content = test_err(
            400,
            r#"{"error":{"message":"message text contains an image special token"}}"#,
        );
        assert_eq!(classify_err(&content), ErrorClass::ContentRejection);
        let content_200 = test_err(
            200,
            r#"{"error":{"message":"Input image data may contain inappropriate content.","code":"data_inspection_failed"}}"#,
        );
        assert_eq!(classify_err(&content_200), ErrorClass::ContentRejection);
        let guardrail = test_err(403, r#"{"error":{"message":"blocked by guardrail"}}"#);
        assert_eq!(classify_err(&guardrail), ErrorClass::Guardrail);
        // A refusal is terminal whatever its status: an upstream 5xx is the only exception, so an
        // outage that mentions moderation keeps retrying.
        assert_eq!(
            classify_err(&test_err(503, "upstream moderation service unavailable")),
            ErrorClass::Retryable
        );
        // A plain client error is neither of the two.
        assert_eq!(
            classify_err(&test_err(400, r#"{"error":{"message":"bad tool schema"}}"#)),
            ErrorClass::InvalidRequest
        );
    }

    /// A 429 is a rate limit whatever its body says: its body quotes upstream metadata that can
    /// contain refusal wording, so the terminal vocabulary is not consulted for it. The billing and
    /// quota hints are the only body text that decides a 429.
    #[test]
    fn a_429_stays_retryable_whatever_refusal_wording_its_body_carries() {
        for body in [
            "rate limit exceeded: content policy",
            "429 blocked by moderation, retry later",
            "upstream refusal: too many requests",
        ] {
            assert_eq!(
                classify_err(&test_err(429, body)),
                ErrorClass::Retryable,
                "a 429 must stay retryable for body '{body}'"
            );
        }
    }

    /// A body that merely mentions a refusal is not a refusal: a proxy that fails to serve says
    /// "upstream refusal", and a moderation *service* can be unavailable. Only a content rejection
    /// is inferred from the wording of a code-less 2xx body, so these stay retryable.
    #[test]
    fn a_merely_refusal_flavoured_body_stays_retryable() {
        for body in [
            "upstream refusal",
            "moderation service unavailable",
            "content_filter backend timed out",
        ] {
            let status = implied_status(body);
            assert_eq!(status, 502, "{body}");
            assert_eq!(
                classify_err(&test_err(status, body)),
                ErrorClass::Retryable,
                "{body}"
            );
        }
    }

    /// A provider error envelope inside a 2xx body carries no status of its own; the implied status
    /// decides whether the retry loop keeps going. A content rejection means the request will be
    /// refused again (400), anything else is an output-production failure (502).
    #[test]
    fn an_in_body_error_implies_a_terminal_status_only_for_a_content_rejection() {
        assert_eq!(implied_status("text contains a special token"), 400);
        assert_eq!(implied_status("blocked by guardrail"), 502);
        assert_eq!(implied_status("upstream provider is overloaded"), 502);
        assert_eq!(implied_status(""), 502);
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
        // A 408 is a timeout whatever its body says: the wording step does not run for it.
        assert_eq!(
            classify_err(&test_err(408, r#"{"error":{"message":"invalid request"}}"#)),
            ErrorClass::Retryable
        );

        // OpenRouter 502 "invalid response" → NOT InvalidRequest: the word "invalid" alone does not
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
                matches!(classify_err(&err), ErrorClass::InvalidRequest),
                "hint '{hint}' should be classified as InvalidRequest"
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
        // A genuine quota-exhaustion 429 (no retryable override signal) stays InvalidRequest — the
        // fix must not break existing billing detection.
        let body = r#"{"error":{"code":429,"message":"You have exceeded your quota. insufficient_quota"}}"#;
        let err = test_err(429, body);
        assert!(
            matches!(classify_err(&err), ErrorClass::InvalidRequest),
            "genuine quota exhaustion 429 without override should remain InvalidRequest"
        );
    }

    #[test]
    fn retryable_overrides_do_not_affect_non_429() {
        // Retryable override signals outside the 429 context change nothing — a 400 with
        // "temporarily rate-limited" is still a client error (InvalidRequest by status code).
        let err = test_err(400, "Bad Request: temporarily rate-limited");
        assert!(
            matches!(classify_err(&err), ErrorClass::InvalidRequest),
            "400 with override text should still be InvalidRequest"
        );
    }
}
