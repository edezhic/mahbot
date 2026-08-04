use super::Provider;
use crate::util::error::HttpError;
use crate::{ChatRequest, ChatResponse};
use async_trait::async_trait;
use std::time::Duration;

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

impl ErrorClass {
    pub(crate) const fn reason_label(self) -> &'static str {
        match self {
            Self::Retryable => "retryable",
            Self::NonRetryable => "non_retryable",
        }
    }
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

/// Try to extract a Retry-After value (in milliseconds) from an error.
///
/// Extracts from the typed [`HttpError::retry_after_ms`] field when the
/// error wraps a [`HttpError`]. Returns `None` for non-structured errors
/// (transport errors, JSON parse errors, etc.) since those never carry a
/// Retry-After value.
///
/// **Note for future providers**: if a new [`Provider`] implementation returns
/// errors with Retry-After information that do NOT wrap [`HttpError`],
/// a string-based fallback path may need to be added here.
pub(crate) fn parse_retry_after_ms(err: &anyhow::Error) -> Option<u64> {
    // ── Typed path: extract from structured HttpError ──
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        return http_err.retry_after_ms;
    }
    None
}

// ── Resilient Provider Wrapper ────────────────────────────────────────────
// Retry loop with exponential backoff, respecting Retry-After headers.
// Loop invariant: `failures` accumulates every failed attempt so the final
// error message gives operators a complete diagnostic trail.

/// Provider wrapper with retry logic.
pub(crate) struct ReliableProvider {
    name: String,
    provider: Box<dyn Provider>,
    max_retries: u32,
    base_backoff_ms: u64,
}

impl ReliableProvider {
    #[must_use]
    pub fn new(
        name: String,
        provider: Box<dyn Provider>,
        max_retries: u32,
        base_backoff_ms: u64,
    ) -> Self {
        Self {
            name,
            provider,
            max_retries,
            base_backoff_ms: base_backoff_ms.max(50),
        }
    }

    /// Compute backoff duration, respecting Retry-After if present.
    /// When no Retry-After header exists, jitter is applied within
    /// ±25% of base to prevent thundering herd when multiple agents
    /// retry simultaneously on transient errors (5xx, timeouts, etc.).
    pub(crate) fn compute_backoff(base: u64, err: &anyhow::Error) -> u64 {
        if let Some(retry_after) = parse_retry_after_ms(err) {
            // Retry-After is authoritative — follow it precisely,
            // clamped to [base, RETRY_AFTER_MAX_MS] (unified with the
            // scoped retry paths in src/retry.rs).
            retry_after.min(crate::retry::RETRY_AFTER_MAX_MS).max(base)
        } else {
            // Jitter: randomize within [75%, 125%) of base so parallel agents
            // retrying on the same transient error don't synchronize.
            let half_range = base / 2;

            base - base / 4 + (rand::random::<u64>() % half_range)
        }
    }
}

#[async_trait]
impl Provider for ReliableProvider {
    async fn warmup(&self) -> anyhow::Result<()> {
        self.provider.warmup().await
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let mut failures = Vec::new();
        let mut backoff_ms = self.base_backoff_ms;

        for attempt in 0..=self.max_retries {
            match self.provider.chat(request.clone()).await {
                Ok(resp) => {
                    if attempt > 0 {
                        tracing::info!(
                            provider = self.name,
                            attempt,
                            "Provider recovered after retry"
                        );
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    let class = classify_err(&e);
                    let error_detail = e.to_string();
                    let reason = class.reason_label();

                    failures.push(format!(
                        "provider={} attempt {}/{}: {}; error={}",
                        self.name,
                        attempt + 1,
                        self.max_retries + 1,
                        reason,
                        error_detail,
                    ));

                    let can_retry = class == ErrorClass::Retryable;

                    // When a 429 body doesn't match any known non-retryable
                    // hint, classify_err falls through to Retryable silently.
                    // Log the body at debug so operators can detect provider-side
                    // error-format changes (e.g., "quota_exhausted" → "credit_limit_reached").
                    if can_retry
                        && let Some(http_err) = e.downcast_ref::<HttpError>()
                        && http_err.status == 429
                    {
                        tracing::debug!(
                            provider = self.name,
                            status = http_err.status,
                            body = %http_err.body,
                            "HTTP 429 body did not match any non-retryable \
                             hint — treating as retryable"
                        );
                    }

                    if can_retry && attempt < self.max_retries {
                        let wait = Self::compute_backoff(backoff_ms, &e);

                        // sleep_or_shutdown returns false immediately if the
                        // global shutdown token is already cancelled, or when
                        // it fires during sleep — no separate pre-check needed.
                        if !crate::shutdown::sleep_or_shutdown(Duration::from_millis(wait)).await {
                            tracing::info!(
                                provider = self.name,
                                attempt = attempt + 1,
                                "Provider shutting down — aborting retry loop"
                            );
                            break;
                        }

                        tracing::warn!(
                            provider = self.name,
                            attempt = attempt + 1,
                            reason,
                            error = %error_detail,
                            "Provider call failed, retrying"
                        );
                        // Doubling is capped at the unified 60 s backoff cap
                        // (same value as the scoped paths' default max backoff).
                        backoff_ms = backoff_ms
                            .saturating_mul(2)
                            .min(crate::retry::DEFAULT_RETRY_MAX_BACKOFF_MS);
                    } else {
                        let log_msg = match class {
                            ErrorClass::NonRetryable => "Non-retryable error, aborting",
                            ErrorClass::Retryable => "Exhausted retries",
                        };
                        tracing::warn!(
                            provider = self.name,
                            attempt = attempt + 1,
                            reason,
                            error = %error_detail,
                            "{log_msg}"
                        );
                        break;
                    }
                }
            }
        }

        anyhow::bail!("All attempts failed.\n{}", failures.join("\n"))
    }

    /// Scoped single-attempt chat (mahbot-1066): delegates to the inner
    /// provider, bypassing this wrapper's retry loop — the outer retry loops
    /// in [`crate::retry`] are the single retry authority for scoped calls.
    async fn chat_scoped(
        &self,
        request: ChatRequest,
        idle_timeout: std::time::Duration,
        deadline: std::time::Instant,
    ) -> Result<ChatResponse, super::ScopedCallError> {
        self.provider
            .chat_scoped(request, idle_timeout, deadline)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatMessage;
    use crate::providers::test_request;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Wrapper around [`HttpError::new`] that sets context="test" and
    /// retry_after=None, reducing boilerplate in error-classification tests.
    fn test_err(status: u16, body: &str) -> anyhow::Error {
        anyhow::Error::from(HttpError::new(status, "test", body, None))
    }

    /// Unified test mock. Covers all failure modes: simple retry gating,
    /// model-specific failures, context overflow, upstream rate limits with
    /// embedded billing codes, and native tool calls.
    struct TestProvider {
        calls: Arc<AtomicUsize>,
        fail_until_attempt: usize,
        response_text: &'static str,
        error: &'static str,
        context_overflow: bool,
        tool_schema_error: bool,
        /// When true, returns an HTTP 429 whose body mimics an OpenRouter proxy
        /// forwarding a transient upstream rate limit — the body contains both
        /// a retryable override signal ("temporarily rate-limited") and a
        /// non-retryable hint ("insufficient_quota") in metadata, which
        /// exercises the RETRYABLE_OVERRIDES classifier logic.
        upstream_rate_limit: bool,
        tool_calls: Vec<crate::ToolCall>,
        warmup_fails: bool,
    }

    impl TestProvider {
        fn new(response_text: &'static str) -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                fail_until_attempt: 0,
                response_text,
                error: "mock error",
                context_overflow: false,
                tool_schema_error: false,
                upstream_rate_limit: false,
                tool_calls: Vec::new(),
                warmup_fails: false,
            }
        }

        fn with_fail(mut self, until_attempt: usize, error: &'static str) -> Self {
            self.fail_until_attempt = until_attempt;
            self.error = error;
            self
        }

        fn with_context_overflow(mut self, fail_until: usize) -> Self {
            self.context_overflow = true;
            self.fail_until_attempt = fail_until;
            self
        }

        fn with_tool_schema_error(mut self, fail_until: usize) -> Self {
            self.tool_schema_error = true;
            self.fail_until_attempt = fail_until;
            self
        }

        fn with_upstream_rate_limit(mut self, fail_until: usize) -> Self {
            self.upstream_rate_limit = true;
            self.fail_until_attempt = fail_until;
            self
        }

        fn with_calls(mut self, calls: Arc<AtomicUsize>) -> Self {
            self.calls = calls;
            self
        }

        fn with_warmup_fail(mut self) -> Self {
            self.warmup_fails = true;
            self
        }

        fn make_error(&self) -> String {
            if self.context_overflow {
                "request (8968 tokens) exceeds the available context size (8448 tokens), try increasing it".to_string()
            } else if self.tool_schema_error {
                "tool call validation failed: attempted to call tool 'recall' which was not in request".to_string()
            } else if self.upstream_rate_limit {
                // Simulate OpenRouter forwarding a transient upstream rate limit.
                // The body contains both a retryable override signal
                // ("temporarily rate-limited") and a non-retryable billing hint
                // ("insufficient_quota" in metadata.provider_error_code).
                r#"{"error":{"code":429,"message":"Provider returned error","metadata":{"provider_name":"Qwen","provider_error_code":"insufficient_quota","raw":"upstream error: temporarily rate-limited upstream, please retry"}}}"#.to_string()
            } else {
                self.error.to_string()
            }
        }

        fn check_fail(&self, attempt: usize) -> bool {
            attempt <= self.fail_until_attempt
        }
    }

    #[async_trait]
    impl Provider for TestProvider {
        async fn chat(&self, _request: ChatRequest) -> anyhow::Result<ChatResponse> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);

            if self.check_fail(call + 1) {
                // Context-overflow and tool-schema errors reach classify_err
                // via HttpError with status 400, so they are correctly classified
                // as NonRetryable by the status-code check (step 3).
                if self.context_overflow {
                    return Err(test_err(400, &self.make_error()));
                }
                if self.tool_schema_error {
                    return Err(test_err(400, &self.make_error()));
                }
                // Upstream rate-limit errors reach classify_err via HttpError
                // with status 429, exercising the RETRYABLE_OVERRIDES logic.
                if self.upstream_rate_limit {
                    return Err(test_err(429, &self.make_error()));
                }
                anyhow::bail!("{}", self.make_error());
            }

            Ok(ChatResponse {
                text: Some(self.response_text.to_string()),
                tool_calls: self.tool_calls.clone(),
                ..Default::default()
            })
        }

        async fn warmup(&self) -> anyhow::Result<()> {
            if self.warmup_fails {
                anyhow::bail!("warmup failed");
            }
            Ok(())
        }
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

    #[tokio::test]
    async fn chat_retries_then_recovers() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = ReliableProvider::new(
            "primary".into(),
            Box::new(
                TestProvider::new("history ok")
                    .with_fail(1, "temporary")
                    .with_calls(calls.clone()),
            ) as Box<dyn Provider>,
            2,
            50,
        );

        let messages = vec![ChatMessage::system("system"), ChatMessage::user("hello")];
        let result = provider
            .chat(test_request(messages.clone(), None))
            .await
            .unwrap();
        assert_eq!(result.text.as_deref(), Some("history ok"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    // ── Retry-After parsing ──

    #[test]
    fn backoff_and_retry_after() {
        // ── parse_retry_after_ms unit tests ──
        let with_retry = HttpError::new(429, "test", "rate limited", Some(5000));
        assert_eq!(
            parse_retry_after_ms(&anyhow::Error::from(with_retry)),
            Some(5000)
        );

        let no_retry = test_err(429, "rate limit");
        assert_eq!(parse_retry_after_ms(&no_retry), None);

        // ── compute_backoff: respects retry-after ──
        let structured =
            anyhow::Error::from(HttpError::new(429, "test", "rate limited", Some(3_000)));
        assert_eq!(ReliableProvider::compute_backoff(500, &structured), 3_000);

        // ── compute_backoff: clamps retry-after to MAX_BACKOFF (60s) ──
        let with_long_retry =
            anyhow::Error::from(HttpError::new(429, "test", "rate limit", Some(120_000)));
        assert_eq!(
            ReliableProvider::compute_backoff(500, &with_long_retry),
            60_000
        );

        // ── compute_backoff: jittered fallback when no retry-after ──
        let no_header = test_err(500, "error");
        let backoff = ReliableProvider::compute_backoff(500, &no_header);
        assert!(
            (375..625).contains(&backoff),
            "expected backoff in [375, 625), got {backoff}"
        );
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

    #[tokio::test]
    async fn chat_returns_aggregated_error_when_all_retries_exhausted() {
        let provider = ReliableProvider::new(
            "p1".into(),
            Box::new(TestProvider::new("never").with_fail(usize::MAX, "p1 chat error"))
                as Box<dyn Provider>,
            0,
            1,
        );

        let messages = vec![ChatMessage::user("hello")];
        let request = test_request(messages.clone(), None);
        let err = provider
            .chat(request)
            .await
            .expect_err("all attempts should fail");
        let msg = err.to_string();
        assert!(msg.contains("All attempts failed"));
        assert!(msg.contains("provider=p1"));
        assert!(msg.contains("error=p1 chat error"));
        assert!(msg.contains("retryable"));
    }

    #[tokio::test]
    async fn warmup_propagates_inner_error() {
        let inner = TestProvider::new("unused").with_warmup_fail();
        let provider =
            ReliableProvider::new("test".into(), Box::new(inner) as Box<dyn Provider>, 0, 1);
        let err = provider
            .warmup()
            .await
            .expect_err("warmup should propagate error");
        assert!(
            err.to_string().contains("warmup failed"),
            "expected 'warmup failed', got: {err}"
        );
    }

    #[tokio::test]
    async fn warmup_ok_when_inner_succeeds() {
        let inner = TestProvider::new("ok");
        let provider =
            ReliableProvider::new("test".into(), Box::new(inner) as Box<dyn Provider>, 0, 1);
        provider.warmup().await.expect("warmup should succeed");
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

    #[tokio::test]
    async fn chat_context_window_exceeded_is_not_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = ReliableProvider::new(
            "primary".into(),
            Box::new(
                TestProvider::new("ok after overflow")
                    .with_context_overflow(2)
                    .with_calls(calls.clone()),
            ) as Box<dyn Provider>,
            3,
            1,
        );

        let messages = vec![ChatMessage::user("test")];
        let result = provider.chat(test_request(messages.clone(), None)).await;
        assert!(
            result.is_err(),
            "context window errors are non-retryable, should fail immediately"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "should not retry context overflow"
        );
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

    #[tokio::test]
    async fn chat_upstream_rate_limit_is_retried() {
        // Integration test: a provider returning HTTP 429 with both
        // "insufficient_quota" (in metadata) and "temporarily rate-limited"
        // (in raw upstream text) must be retried, not aborted at attempt 1.
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = ReliableProvider::new(
            "primary".into(),
            Box::new(
                TestProvider::new("ok after upstream rate limit")
                    .with_upstream_rate_limit(1)
                    .with_calls(calls.clone()),
            ) as Box<dyn Provider>,
            2,
            1,
        );

        let messages = vec![ChatMessage::user("test")];
        let result = provider.chat(test_request(messages.clone(), None)).await;
        assert!(
            result.is_ok(),
            "upstream rate-limit 429 with retryable override should recover after retry"
        );
        assert_eq!(
            result.unwrap().text.as_deref(),
            Some("ok after upstream rate limit"),
        );
        // Should have failed once (upstream rate limit) then recovered
        assert_eq!(calls.load(Ordering::SeqCst), 2);
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

    #[tokio::test]
    async fn chat_tool_schema_error_is_not_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = ReliableProvider::new(
            "primary".into(),
            Box::new(
                TestProvider::new("unused")
                    .with_tool_schema_error(10)
                    .with_calls(calls.clone()),
            ) as Box<dyn Provider>,
            3,
            1,
        );

        let messages = vec![ChatMessage::user("test")];
        let result = provider.chat(test_request(messages.clone(), None)).await;
        assert!(
            result.is_err(),
            "tool schema errors are non-retryable, should fail immediately"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "should not retry tool schema errors"
        );
    }
}
