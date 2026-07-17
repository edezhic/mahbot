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

/// Extended error classification for transcriber fallback chains.
///
/// Unlike [`ErrorClass`] (binary Retryable vs NonRetryable), this variant
/// distinguishes *model-specific* failures (which should fall through to
/// the next model in the chain) from *global* failures (billing, auth —
/// abort the entire chain).  Transient errors (rate limits, 5xx, network)
/// remain retryable on the same model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TranscriberErrorClass {
    /// Transient error — retry the current model with backoff.
    RetryModel,
    /// Model-specific error (404 not found, 400 unsupported) — skip to next model.
    SkipModel,
    /// Fatal/global error (auth, billing/quota exhausted) — abort entire chain.
    AbortChain,
}

impl TranscriberErrorClass {
    pub(crate) const fn reason_label(self) -> &'static str {
        match self {
            Self::RetryModel => "retry_model",
            Self::SkipModel => "skip_model",
            Self::AbortChain => "abort_chain",
        }
    }
}

/// Body-text hints that indicate permanent (non-retryable) errors when the
/// HTTP status-code check is ambiguous — specifically, HTTP 429 Too Many
/// Requests is excluded from the status-based classification (rate limits
/// are transient), so these billing/quota hints override 429 to prevent
/// endless retries on exhausted accounts.
///
/// All other non-retryable errors (context window exceeded, tool schema
/// validation, auth failures) are reliably caught by the HTTP 4xx status-code
/// check (step 2 in [`classify_err`]) and do NOT need entries here.  That
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
/// 1. **Billing/quota body-text hints** — The [`NON_RETRYABLE_HINTS`] entries
///    override the default Retryable classification for HTTP 429 responses
///    (quota exhaustion is permanent, not transient).
/// 2. **4xx status codes** (except 408 Request Timeout and 429 Too Many Requests)
///    — structured [`HttpError`] downcast.
/// 3. Default to [`Retryable`](ErrorClass::Retryable).
pub(crate) fn classify_err(err: &anyhow::Error) -> ErrorClass {
    // ── Typed path: use structured fields from HttpError directly ──
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        // Body-text hints indicate permanent errors — only relevant when
        // the status code is ambiguous (HTTP 429 is normally retryable,
        // but billing/quota exhaustion is permanent).
        if http_err.status == 429 {
            let body_lower = http_err.body.to_lowercase();
            if NON_RETRYABLE_HINTS.iter().any(|h| body_lower.contains(h)) {
                return ErrorClass::NonRetryable;
            }
        }
        // 4xx codes (except 408 Request Timeout and 429 Too Many Requests)
        if (400..500).contains(&http_err.status) && http_err.status != 408 && http_err.status != 429
        {
            return ErrorClass::NonRetryable;
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

/// Classify an error for the transcriber fallback chain.
///
/// Returns one of three classes:
/// * [`RetryModel`](TranscriberErrorClass::RetryModel) — transient, retry the current model.
/// * [`SkipModel`](TranscriberErrorClass::SkipModel) — model-specific, try the next model.
/// * [`AbortChain`](TranscriberErrorClass::AbortChain) — fatal (auth, billing), abort all.
///
/// The distinction from [`classify_err`] is that model-specific 4xx errors
/// (400, 404, 403, etc.) are classified as [`SkipModel`] rather than
/// [`NonRetryable`](ErrorClass::NonRetryable), so the fallback chain can
/// try the next model instead of aborting.
#[must_use]
pub(crate) fn classify_transcriber_err(err: &anyhow::Error) -> TranscriberErrorClass {
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        match http_err.status {
            // 401 Unauthorized or 402 Payment Required → abort chain.
            401 | 402 => return TranscriberErrorClass::AbortChain,
            // 429: billing/quota hints → abort; otherwise retry.
            429 => {
                let body_lower = http_err.body.to_lowercase();
                if NON_RETRYABLE_HINTS.iter().any(|h| body_lower.contains(h)) {
                    return TranscriberErrorClass::AbortChain;
                }
                return TranscriberErrorClass::RetryModel;
            }
            // Model-specific 4xx (400, 403, 404, etc.) → skip to next model.
            400..=499 => return TranscriberErrorClass::SkipModel,
            // 5xx → transient, retry.
            _ => return TranscriberErrorClass::RetryModel,
        }
    }
    // Non-HTTP errors (timeouts, DNS, etc.) → transient, retry.
    TranscriberErrorClass::RetryModel
}

// ── Resilient Provider Wrapper ────────────────────────────────────────────
// Retry loop with exponential backoff, respecting Retry-After headers.
// Loop invariant: `failures` accumulates every failed attempt so the final
// error message gives operators a complete diagnostic trail.

/// Provider wrapper with retry logic.
pub struct ReliableProvider {
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
            // clamped to [base, 30_000] ms.
            retry_after.min(30_000).max(base)
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
                        backoff_ms = backoff_ms.saturating_mul(2);
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
    /// model-specific failures, context overflow, and native tool calls.
    struct TestProvider {
        calls: Arc<AtomicUsize>,
        fail_until_attempt: usize,
        response_text: &'static str,
        error: &'static str,
        context_overflow: bool,
        tool_schema_error: bool,
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
                // as NonRetryable by the status-code check (step 2).
                if self.context_overflow {
                    return Err(test_err(400, &self.make_error()));
                }
                if self.tool_schema_error {
                    return Err(test_err(400, &self.make_error()));
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

        // ── compute_backoff: clamps retry-after to MAX_BACKOFF (30s) ──
        let with_long_retry =
            anyhow::Error::from(HttpError::new(429, "test", "rate limit", Some(120_000)));
        assert_eq!(
            ReliableProvider::compute_backoff(500, &with_long_retry),
            30_000
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

    // ── Transcriber error classification tests ───────────────

    #[test]
    fn transcriber_err_abort_chain() {
        // 401 Unauthorized → abort chain (auth failure)
        assert!(
            matches!(
                classify_transcriber_err(&test_err(401, "Unauthorized")),
                TranscriberErrorClass::AbortChain
            ),
            "401 should abort chain"
        );
        // 402 Payment Required → abort chain
        assert!(
            matches!(
                classify_transcriber_err(&test_err(402, "Payment Required")),
                TranscriberErrorClass::AbortChain
            ),
            "402 should abort chain"
        );
        // 429 with billing/quota hints → abort chain
        for hint in NON_RETRYABLE_HINTS {
            let err = test_err(429, hint);
            assert!(
                matches!(
                    classify_transcriber_err(&err),
                    TranscriberErrorClass::AbortChain
                ),
                "429 with hint '{hint}' should abort chain"
            );
        }
    }

    #[test]
    fn transcriber_err_skip_model() {
        // 400 Bad Request (model unsupported) → skip to next model
        assert!(
            matches!(
                classify_transcriber_err(&test_err(400, "model not supported")),
                TranscriberErrorClass::SkipModel
            ),
            "400 should skip model"
        );
        // 403 Forbidden (model access denied) → skip to next model
        assert!(
            matches!(
                classify_transcriber_err(&test_err(403, "Forbidden")),
                TranscriberErrorClass::SkipModel
            ),
            "403 should skip model"
        );
        // 404 Not Found (model not found) → skip to next model
        assert!(
            matches!(
                classify_transcriber_err(&test_err(404, "model not found")),
                TranscriberErrorClass::SkipModel
            ),
            "404 should skip model"
        );
        // 408 Request Timeout → skip to next model (unlike classify_err which
        // retries, the transcriber takes a more aggressive fallback approach)
        assert!(
            matches!(
                classify_transcriber_err(&test_err(408, "Request Timeout")),
                TranscriberErrorClass::SkipModel
            ),
            "408 should skip model in transcriber"
        );
        // 405 Method Not Allowed → skip to next model
        assert!(
            matches!(
                classify_transcriber_err(&test_err(405, "Method Not Allowed")),
                TranscriberErrorClass::SkipModel
            ),
            "405 should skip model"
        );
    }

    #[test]
    fn transcriber_err_retry_model() {
        // 429 rate limit (without billing hints) → retry current model
        assert!(
            matches!(
                classify_transcriber_err(&test_err(429, "Too Many Requests")),
                TranscriberErrorClass::RetryModel
            ),
            "429 without billing hints should retry"
        );
        assert!(
            matches!(
                classify_transcriber_err(&test_err(429, "rate limit exceeded")),
                TranscriberErrorClass::RetryModel
            ),
            "429 without billing hints should retry"
        );
        // 500 Internal Server Error → retry
        assert!(
            matches!(
                classify_transcriber_err(&test_err(500, "Internal Server Error")),
                TranscriberErrorClass::RetryModel
            ),
            "500 should retry"
        );
        // 502 Bad Gateway → retry
        assert!(
            matches!(
                classify_transcriber_err(&test_err(502, "Bad Gateway")),
                TranscriberErrorClass::RetryModel
            ),
            "502 should retry"
        );
        // 503 Service Unavailable → retry
        assert!(
            matches!(
                classify_transcriber_err(&test_err(503, "Service Unavailable")),
                TranscriberErrorClass::RetryModel
            ),
            "503 should retry"
        );
        // Non-HTTP errors (timeouts, DNS) → retry
        assert!(
            matches!(
                classify_transcriber_err(&anyhow::anyhow!("connection reset")),
                TranscriberErrorClass::RetryModel
            ),
            "non-HTTP errors should retry"
        );
        assert!(
            matches!(
                classify_transcriber_err(&anyhow::anyhow!("timeout")),
                TranscriberErrorClass::RetryModel
            ),
            "non-HTTP errors should retry"
        );
    }

    #[test]
    fn transcriber_err_5xx_with_hint_text_is_retryable() {
        // Regression: 5xx responses from proxy providers (e.g. OpenRouter
        // forwarding an upstream error) may contain billing/quota language
        // in the body. These must remain RetryModel — the issue is a
        // transient upstream failure, not account exhaustion.
        for hint in NON_RETRYABLE_HINTS {
            let err = test_err(502, &format!("upstream error: {hint}"));
            assert!(
                matches!(
                    classify_transcriber_err(&err),
                    TranscriberErrorClass::RetryModel
                ),
                "502 with hint '{hint}' should remain RetryModel"
            );
        }
    }
}
