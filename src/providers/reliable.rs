use super::Provider;
use crate::util::error::HttpError;
use crate::util::http::extract_http_status;
use crate::{ChatRequest, ChatResponse, StreamEvent, StreamResult};
use async_trait::async_trait;
use futures_util::stream;
use std::time::Duration;

// reqwest is used for typed downcast in classify_err — not for direct HTTP calls.
use reqwest;

// ── Error Classification ─────────────────────────────────────────────────
// Errors are split into retryable (transient server/network failures) and
// non-retryable (permanent client errors). This distinction drives whether
// the retry loop continues or aborts immediately — avoiding wasted latency
// on errors that cannot self-heal.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorClass {
    /// A transient error that may resolve with retries (timeouts, 5xx, etc.).
    Retryable,
    /// A non-retryable client error (auth, invalid model, billing/quota exhausted,
    /// tool schema validation failure, etc.).
    NonRetryable,
}

impl ErrorClass {
    const fn reason_label(self) -> &'static str {
        match self {
            Self::Retryable => "retryable",
            Self::NonRetryable => "non_retryable",
        }
    }
}

/// Body-text hints that indicate permanent (non-retryable) errors
/// regardless of HTTP status code — checked by [`classify_by_status_code`]
/// before status-based classification.
const NON_RETRYABLE_HINTS: &[&str] = &[
    "exceeds the context window",
    "exceeds the available context size",
    "context window of this model",
    "maximum context length",
    "context length exceeded",
    "too many tokens",
    "token limit exceeded",
    "prompt is too long",
    "input is too long",
    "prompt exceeds max length",
    "tool call validation failed",
    "which was not in request",
    "not found in tool list",
    "invalid_tool_call",
    "invalid api key",
    "incorrect api key",
    "missing api key",
    "api key not set",
    "authentication failed",
    "auth failed",
    "unauthorized",
    "forbidden",
    "permission denied",
    "access denied",
    "invalid token",
    "insufficient balance",
    "insufficient_quota",
    "quota exhausted",
    "quota exceeded",
    "error code 1113",
];

/// Fallback classification reached when no body-text overrides or recognized
/// 4xx status code apply. Checks model-not-found patterns, falling through
/// to a default of retryable.
///
/// This function is only called from [`classify_by_status_code`] after all
/// non-retryable body-text hints have already been checked and returned no
/// match. It handles the model-not-found composite pattern and other
/// non-status-code signals.
fn classify_fallback(lower: &str) -> ErrorClass {
    // Model not found — composite check to catch variants like
    // "model 'xyz' is unknown" alongside "model unknown".
    if lower.contains("model")
        && (lower.contains("not found")
            || lower.contains("unknown")
            || lower.contains("unsupported")
            || lower.contains("does not exist"))
    {
        return ErrorClass::NonRetryable;
    }
    ErrorClass::Retryable
}

/// Dispatch by HTTP status code using [`is_non_retryable_4xx`].
///
/// Body-text hints for permanent (non-retryable) errors are checked **before**
/// status-based classification — they indicate permanent errors regardless of
/// HTTP status code (e.g., a 429 with context-window body text is non-retryable,
/// not a transient rate limit).
///
/// Accepts an optional status code parsed from the error:
/// - `None` → falls through to [`classify_fallback`]
/// - `Some(4xx)` where [`is_non_retryable_4xx`] returns `true` → [`NonRetryable`](ErrorClass::NonRetryable)
/// - any other status (408, 429, 5xx, 3xx, etc.) → falls through to [`classify_fallback`]
#[inline]
fn classify_by_status_code(status: Option<u16>, lower: &str) -> ErrorClass {
    // Non-retryable body-text hints are checked before status-based
    // classification — they indicate permanent errors regardless of
    // HTTP status code.
    if NON_RETRYABLE_HINTS.iter().any(|h| lower.contains(h)) {
        return ErrorClass::NonRetryable;
    }
    if status.is_some_and(is_non_retryable_4xx) {
        ErrorClass::NonRetryable
    } else {
        classify_fallback(lower)
    }
}

/// Returns `true` for 4xx status codes that are NOT retryable.
///
/// 408 (Request Timeout) and 429 (Too Many Requests) are excluded —
/// both are transient and retried with appropriate backoff.
/// Transient 429s are classified as retryable; 429s with
/// billing/quota body signals remain non-retryable via
/// body-text checks in [`classify_by_status_code`].
fn is_non_retryable_4xx(code: u16) -> bool {
    (400..500).contains(&code) && code != 408 && code != 429
}

/// Classify an error into one of the [`ErrorClass`] variants.
///
/// ## Cascade Order
/// 1. **Typed path** (downcast to [`HttpError`] succeeds): dispatch on
///    structured status code via [`classify_by_status_code`] — which checks
///    non-retryable body-text hints, then 4xx codes other than
///    408 and 429 as [`NonRetryable`](ErrorClass::NonRetryable), else
///    [`classify_fallback`]
/// 2. **Transport typed path** (downcast to [`reqwest::Error`] succeeds): dispatch
///    using typed `is_timeout()`, `is_connect()`, `is_builder()`, `is_redirect()`,
///    `is_status()` via [`classify_transport_err`] — avoids string-matching transport
///    error messages
/// 3. **String-fallback path** (no structured wrapper): extract HTTP status from
///    string via [`crate::util::http::extract_http_status`], then dispatch via
///    [`classify_by_status_code`]
fn classify_err(err: &anyhow::Error) -> ErrorClass {
    let msg = err.to_string();
    let lower = msg.to_lowercase();

    // ── Typed path: extract from structured HttpError ──
    if let Some(http_err) = err.downcast_ref::<HttpError>() {
        return classify_by_status_code(Some(http_err.status), &lower);
    }

    // ── Transport error typed path: extract from reqwest::Error ──
    if let Some(transport_err) = err.downcast_ref::<reqwest::Error>() {
        return classify_transport_err(transport_err, &lower);
    }

    // ── Fallback: string-parsing path (for non-structured errors) ──
    classify_by_status_code(extract_http_status(&msg), &lower)
}

/// Classify a transport error using typed `reqwest::Error` properties.
///
/// Uses `is_timeout()`, `is_connect()`, `is_builder()`, `is_redirect()`,
/// and `is_status()` with `.status()` for precise classification, avoiding
/// the string-matching fallback.
///
/// ## Classification Rules
/// - **Timeout / connect**: `Retryable` — transient network conditions
/// - **Builder / redirect**: `NonRetryable` — misconfiguration, won't self-heal
/// - **Status error** (e.g. 4xx from `error_for_status`): delegate to
///   [`classify_by_status_code`] using the actual status code
/// - **Body / stream errors**: `Retryable` — transient transport issues
fn classify_transport_err(transport_err: &reqwest::Error, lower: &str) -> ErrorClass {
    if transport_err.is_timeout() || transport_err.is_connect() {
        return ErrorClass::Retryable;
    }
    if transport_err.is_builder() || transport_err.is_redirect() {
        return ErrorClass::NonRetryable;
    }
    if transport_err.is_status()
        && let Some(status) = transport_err.status()
    {
        let code = status.as_u16();
        return classify_by_status_code(Some(code), lower);
    }
    // Body read errors, stream errors, default → retryable
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
fn parse_retry_after_ms(err: &anyhow::Error) -> Option<u64> {
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
    fn compute_backoff(base: u64, err: &anyhow::Error) -> u64 {
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

                    if can_retry && attempt < self.max_retries {
                        // Check for global shutdown before sleeping
                        if crate::shutdown::shutdown_token().is_cancelled() {
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
                        let wait = Self::compute_backoff(backoff_ms, &e);
                        if !crate::shutdown::sleep_or_shutdown(Duration::from_millis(wait)).await {
                            break;
                        }
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

    /// Stream a chat request. Streaming errors are not retried because
    /// partial output may have already been delivered to the caller.
    /// When streaming fails, callers (e.g., `agent::llm_call`) typically
    /// fall back to [`chat`](Self::chat), which has full retry logic.
    fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        self.provider.stream_chat(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatMessage, ToolSpec};
    use futures_util::StreamExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
                anyhow::bail!("{}", self.make_error());
            }

            Ok(ChatResponse {
                text: Some(self.response_text.to_string()),
                tool_calls: self.tool_calls.clone(),
                ..Default::default()
            })
        }

        fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
            stream::iter(vec![
                Ok(StreamEvent::ToolCall(crate::ToolCall {
                    id: "call_1".to_string(),
                    name: "shell".to_string(),
                    arguments: serde_json::json!({"command": "date"}),
                })),
                Ok(StreamEvent::Final),
            ])
            .boxed()
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
        // Non-retryable
        assert!(is_non_retryable(&anyhow::anyhow!("401 Unauthorized")));
        assert!(is_non_retryable(&anyhow::anyhow!("403 Forbidden")));
        assert!(is_non_retryable(&anyhow::anyhow!("invalid api key")));
        assert!(is_non_retryable(&anyhow::anyhow!("model not found")));
        assert!(is_non_retryable(&anyhow::anyhow!("model 'xyz' is unknown")));
        assert!(is_non_retryable(&anyhow::anyhow!("insufficient balance")));
        assert!(is_non_retryable(&anyhow::anyhow!("insufficient_quota")));
        assert!(is_non_retryable(&anyhow::anyhow!("quota exhausted")));
        assert!(is_non_retryable(&anyhow::anyhow!("error code 1113")));
        // Retryable
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
            .chat(ChatRequest {
                messages: messages.clone(),
                tools: None,
                model: "test".to_string(),
                allow_image_parts: false,
                temperature: 0.1,
                reasoning_effort: None,
                provider_order: None,
                provider_allow_fallbacks: None,
            })
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

        let no_retry = HttpError::new(429, "test", "rate limit", None);
        assert_eq!(parse_retry_after_ms(&anyhow::Error::from(no_retry)), None);

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
        let no_header = anyhow::Error::from(HttpError::new(500, "test", "error", None));
        let backoff = ReliableProvider::compute_backoff(500, &no_header);
        assert!(
            (375..625).contains(&backoff),
            "expected backoff in [375, 625), got {backoff}"
        );
    }

    #[test]
    fn classify_err_typed_path() {
        // ── HttpError typed path for classify_err ──
        let make_structured = |status: u16, body: &str| -> anyhow::Error {
            anyhow::Error::from(HttpError::new(status, "test", body, None))
        };

        // 429 transient rate limit → retryable (falls through to
        // classify_fallback which returns Retryable for non-billing bodies)
        assert!(matches!(
            classify_err(&make_structured(429, "Too Many Requests")),
            ErrorClass::Retryable
        ));
        assert!(matches!(
            classify_err(&make_structured(429, "rate limit exceeded")),
            ErrorClass::Retryable
        ));

        // 429 with billing/quota body signals → non-retryable
        // (caught by NON_RETRYABLE_HINTS in classify_by_status_code, not by status code)
        assert_eq!(
            classify_err(&make_structured(429, "insufficient balance")),
            ErrorClass::NonRetryable
        );
        assert_eq!(
            classify_err(&make_structured(429, "quota exhausted")),
            ErrorClass::NonRetryable
        );

        // Non-429 4xx → non-retryable
        assert!(matches!(
            classify_err(&make_structured(400, "Bad Request")),
            ErrorClass::NonRetryable
        ));
        assert!(matches!(
            classify_err(&make_structured(403, "Forbidden")),
            ErrorClass::NonRetryable
        ));

        // 408 → fallback (not NonRetryable)
        assert!(matches!(
            classify_err(&make_structured(408, "Request Timeout")),
            ErrorClass::Retryable
        ));

        // 5xx → retryable (fallback)
        assert!(matches!(
            classify_err(&make_structured(500, "Internal Server Error")),
            ErrorClass::Retryable
        ));

        // Context window → NonRetryable (body text analysis)
        assert!(matches!(
            classify_err(&make_structured(
                400,
                "exceeds the context window of this model"
            )),
            ErrorClass::NonRetryable
        ));

        // Tool schema error → NonRetryable (body text analysis)
        assert!(matches!(
            classify_err(&make_structured(400, "tool call validation failed")),
            ErrorClass::NonRetryable
        ));

        // Auth patterns in body → NonRetryable (body-text override in classify_by_status_code)
        assert!(matches!(
            classify_err(&make_structured(403, "unauthorized")),
            ErrorClass::NonRetryable
        ));

        // Model not found → NonRetryable (via is_non_retryable_4xx for 404)
        assert!(matches!(
            classify_err(&make_structured(404, "model not found")),
            ErrorClass::NonRetryable
        ));

        // ZhipuAI billing error code 1113 → NonRetryable
        // (caught by NON_RETRYABLE_HINTS in classify_by_status_code)
        assert_eq!(
            classify_err(&make_structured(429, "error code 1113")),
            ErrorClass::NonRetryable
        );

        // OpenRouter 502 "invalid response" → NOT NonRetryable
        // (the word "invalid" alone does not imply a bad model id)
        assert_eq!(
            classify_err(&make_structured(
                502,
                "Your chosen model is down or we received an invalid response from it"
            )),
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
        let request = ChatRequest {
            messages: messages.clone(),
            tools: None,
            model: "test".to_string(),
            allow_image_parts: false,
            temperature: 0.1,
            reasoning_effort: None,
            provider_order: None,
            provider_allow_fallbacks: None,
        };
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
        // Context window exceeded SHOULD be non-retryable now
        assert!(is_non_retryable(&anyhow::anyhow!(
            "request (8968 tokens) exceeds the available context size (8448 tokens)"
        )));
        assert!(is_non_retryable(&anyhow::anyhow!(
            "This model's maximum context length is 8192 tokens"
        )));
        assert!(is_non_retryable(&anyhow::anyhow!(
            "maximum context length of this model is 128K tokens"
        )));
        // Non-retryable errors should still be non-retryable
        assert!(is_non_retryable(&anyhow::anyhow!("401 Unauthorized")));
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
        let result = provider
            .chat(ChatRequest {
                messages: messages.clone(),
                tools: None,
                model: "test".to_string(),
                allow_image_parts: false,
                temperature: 0.1,
                reasoning_effort: None,
                provider_order: None,
                provider_allow_fallbacks: None,
            })
            .await;
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
        // Detects various tool schema error patterns as NonRetryable
        for msg in [
            r#"Groq API error (400 Bad Request): {"error":{"message":"tool call validation failed: attempted to call tool 'recall' which was not in request"}}"#,
            "tool 'search' which was not in request",
            "function 'foo' not found in tool list",
            "invalid_tool_call: no matching function",
        ] {
            assert!(
                matches!(classify_err(&anyhow::anyhow!("{msg}")), NonRetryable),
                "should detect: {msg}"
            );
        }
        // Pure 400 without tool-schema keywords → also NonRetryable (via is_non_retryable_4xx)
        assert!(
            matches!(
                classify_err(&anyhow::anyhow!(
                    "400 Bad Request: invalid api key provided"
                )),
                NonRetryable
            ),
            "pure 400 should be NonRetryable"
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
        let result = provider
            .chat(ChatRequest {
                messages: messages.clone(),
                tools: None,
                model: "test".to_string(),
                allow_image_parts: false,
                temperature: 0.1,
                reasoning_effort: None,
                provider_order: None,
                provider_allow_fallbacks: None,
            })
            .await;
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

    #[tokio::test]
    async fn stream_chat_works_when_provider_supports_tool_events() {
        let provider =
            ReliableProvider::new("primary".into(), Box::new(TestProvider::new("ok")), 0, 1);

        let request = ChatRequest {
            messages: vec![ChatMessage::user("hello")],
            tools: Some(vec![ToolSpec {
                name: "test".into(),
                description: "A test tool".into(),
                parameters: serde_json::json!({}),
            }]),
            model: "test".to_string(),
            allow_image_parts: false,
            temperature: 0.1,
            reasoning_effort: None,
            provider_order: None,
            provider_allow_fallbacks: None,
        };
        let mut stream = provider.stream_chat(request);
        let first = stream.next().await.unwrap().unwrap();
        if let StreamEvent::ToolCall(tc) = first {
            assert_eq!(tc.name, "shell");
        } else {
            panic!("expected ToolCall event");
        }
    }
}
