//! Structured extraction from conversation history.
//!
//! Provides LLM-powered extraction of structured data (JSON) from agent
//! conversation history, with retry logic.

use serde::de::DeserializeOwned;
use std::time::Instant;

use crate::prompt::load_prompt;
use crate::retry::{FailureClass, RetryFailureRecord, RetryLoop, RetryPolicy};
use crate::util::json::parse_fenced_json;
use crate::{ChatMessage, ChatRequest, ExtractionValidator};

// ── Scoped retry extraction ────────────────────────────────

/// Scoped structured extraction with the hardened outer retry loop
/// ([`crate::retry`] — default 13 attempts / 720 s wall cap). Used by
/// verdict extraction, diagnostics discovery, comment summaries, and
/// research orchestration. `policy_override` supplies a shorter schedule
/// for fail-open comment-only callers ([`crate::retry::RetryPolicy::comment`]
/// — 3 attempts / 90 s); `None` uses the default policy.
///
/// The outer loop is the SINGLE retry authority — provider-internal retries
/// are suppressed via [`crate::providers::chat_scoped`], so total provider
/// HTTP calls per operation are bounded by `max_attempts`.
///
/// Retry semantics per attempt:
/// - Provider/transport/truncation failure → retry byte-identical.
/// - Tool call, unparseable text, or validation failure (`validate` rejects)
///   → treat as parse failure: push raw text + `extraction/retry.md` re-prompt
///   (the ONLY permitted request mutation — extends the cached prefix).
/// - Valid JSON matching `T` and passing `validate` → return immediately.
///
/// `validate` (typically a score ∈ [0,10] check for [`crate::Verdict`]) runs
/// fail-closed: a rejected value is a parse failure, and if all attempts yield
/// rejected values the operation fails with an out-of-range-score
/// classification — a garbage score never passes any gate. It must be
/// `Send + Sync` because extraction runs inside `join_all` futures.
///
/// Terminal failure is [`crate::retry::RetryExhausted`] — it carries the
/// last-attempt raw text (`last_raw`, for verdict-extraction ticket comments) plus
/// the per-attempt diagnostics trail.
#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
pub(crate) async fn retry_extract_structured_scoped<T: DeserializeOwned>(
    history: &[ChatMessage],
    extraction_prompt: &str,
    params: &ChatRequest,
    validate: Option<&ExtractionValidator<T>>,
    policy_override: Option<&RetryPolicy>,
) -> Result<T, crate::retry::RetryExhausted> {
    let mut extraction_history = history.to_vec();
    let operation_started = Instant::now();

    // Only push the extraction prompt if non-empty — caller may have embedded it
    if !extraction_prompt.is_empty() {
        extraction_history.push(ChatMessage::user(extraction_prompt));
    }

    // Telemetry record request: the recorded request derives from params so
    // recorded metadata can never diverge from the bytes actually sent. Only
    // the telemetry fields are read from it — messages are emptied here so the
    // per-attempt spread doesn't clone the caller's history; the one-time
    // params.clone() below is that cost.
    let record_request = ChatRequest {
        messages: vec![],
        ..params.clone()
    };

    let retry_prompt = load_prompt("extraction/retry.md");
    let policy = policy_override
        .cloned()
        .unwrap_or_else(RetryPolicy::current);
    let mut loop_state = RetryLoop::new(&policy, record_request.meta.as_ref());
    let mut last_raw: Option<String> = None;

    for attempt in 1..=policy.max_attempts {
        if loop_state.expired() {
            let exhausted = crate::retry::RetryExhausted::with_last_raw(
                loop_state.into_failures(),
                FailureClass::WallClockExceeded,
                last_raw,
            );
            crate::stats::record_llm_failure(&record_request, operation_started, &exhausted).await;
            return Err(exhausted);
        }

        let attempt_started = Instant::now();
        let request = ChatRequest {
            messages: extraction_history.clone(),
            allow_image_parts: false, // extractions never need image parts
            ..record_request.clone()
        };

        match crate::providers::chat_scoped(request, policy.idle_timeout, loop_state.deadline())
            .await
        {
            Ok(response) => {
                last_raw = Some(response.text_or_empty().to_string());

                // Determine the failure mode: tool call / parse failure /
                // validation rejection all funnel into the re-prompt retry.
                // The fine-grained cause feeds retry-cause telemetry (the
                // parse-class split the JSON-quality tuning needs).
                let (class, detail, cause): (FailureClass, String, &'static str) =
                    if response.tool_calls.is_empty() {
                        let raw = last_raw.as_deref().unwrap_or_default();
                        match parse_fenced_json::<T>(raw) {
                            Ok(result) => {
                                if let Some(validate) = validate
                                    && let Err(msg) = validate(&result)
                                {
                                    (
                                        FailureClass::OutOfRangeScore,
                                        format!("extracted value rejected by validation: {msg}"),
                                        crate::retry::CAUSE_SCORE_VALIDATION,
                                    )
                                } else {
                                    crate::stats::record_llm_success(
                                        &record_request,
                                        operation_started,
                                        attempt,
                                        &response,
                                    )
                                    .await;
                                    return Ok(result);
                                }
                            }
                            Err(e) => (
                                FailureClass::Parse,
                                format!("failed to parse extracted JSON: {e}"),
                                crate::retry::parse_failure_cause(raw),
                            ),
                        }
                    } else {
                        (
                            FailureClass::Parse,
                            "extraction attempt returned a tool call instead of JSON".to_string(),
                            crate::retry::CAUSE_TOOL_CALL,
                        )
                    };

                let err = anyhow::anyhow!("{detail}");
                let rec = RetryFailureRecord::new_simple(
                    attempt,
                    class,
                    &err,
                    attempt_started.elapsed().as_millis() as u64,
                    None,
                )
                .with_cause(cause);
                loop_state.record(attempt, rec).await;

                // Re-prompt: push raw assistant text + retry prompt. Only
                // needed when another attempt will actually run — the final
                // attempt's re-prompt is never sent.
                if attempt < policy.max_attempts {
                    extraction_history
                        .push(ChatMessage::assistant(last_raw.clone().unwrap_or_default()));
                    extraction_history.push(ChatMessage::user(retry_prompt.as_str()));
                }
            }
            Err(scoped_err) => {
                // The final attempt died before producing text (transport /
                // truncation / budget) — clear any earlier text so the
                // ticket comment never labels it 'last attempt'.
                last_raw = None;
                let non_retryable = !scoped_err.class.is_retryable();
                loop_state.record(attempt, scoped_err.record).await;
                if non_retryable {
                    let exhausted = crate::retry::RetryExhausted::with_last_raw(
                        loop_state.into_failures(),
                        scoped_err.class,
                        last_raw,
                    );
                    crate::stats::record_llm_failure(
                        &record_request,
                        operation_started,
                        &exhausted,
                    )
                    .await;
                    return Err(exhausted);
                }
            }
        }

        if let Err(class) = loop_state.sleep_between(attempt).await {
            let exhausted = crate::retry::RetryExhausted::with_last_raw(
                loop_state.into_failures(),
                class,
                last_raw,
            );
            crate::stats::record_llm_failure(&record_request, operation_started, &exhausted).await;
            return Err(exhausted);
        }
    }

    // Exhausted — report the last failure's class so operators see e.g.
    // out-of-range-score (never a garbage pass) rather than a generic class.
    let final_class = loop_state.final_class();
    let exhausted = crate::retry::RetryExhausted::with_last_raw(
        loop_state.into_failures(),
        final_class,
        last_raw,
    );
    crate::stats::record_llm_failure(&record_request, operation_started, &exhausted).await;
    Err(exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::tiny_test_policy;
    use crate::util::test::{FakeProvider, install_fake_provider, retry_tests_lock};
    use crate::{ChatMessage, ChatRequest};
    use std::sync::Arc;

    /// A minimal deserializable target for extraction tests.
    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct FakeVerdict {
        score: u8,
    }

    /// Verdict-style validator: score must be ∈ [0,10].
    fn score_validator(v: &FakeVerdict) -> Result<(), String> {
        if v.score <= 10 {
            Ok(())
        } else {
            Err(format!("score {} out of range", v.score))
        }
    }

    fn test_params() -> ChatRequest {
        ChatRequest {
            messages: vec![],
            tools: None,
            model: "test-model".to_string(),
            allow_image_parts: false,
            max_tokens: None,
            reasoning_effort: None,
            provider_order: None,
            provider_allow_fallbacks: None,
            meta: None,
        }
    }

    fn history() -> Vec<ChatMessage> {
        vec![ChatMessage::user("analyze the ticket")]
    }

    async fn extract_with(
        fake: Arc<FakeProvider>,
    ) -> Result<FakeVerdict, crate::retry::RetryExhausted> {
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _guard = install_fake_provider(provider);
        retry_extract_structured_scoped::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &test_params(),
            Some(&score_validator),
            None,
        )
        .await
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn recovers_after_consecutive_transport_errors() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(
            FakeProvider::new()
                .err(crate::retry::FailureClass::Transport, "transport error 1")
                .err(crate::retry::FailureClass::Transport, "transport error 2")
                .ok(r#"{"score": 8}"#),
        );
        let result = extract_with(fake).await;
        let verdict = result.expect("should recover after 2 transport errors");
        assert_eq!(verdict.score, 8);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn recovers_after_truncated_envelope_parse_errors() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        // Truncated-envelope class (EOF-while-parsing defect class) retried
        // byte-identical, then a full body parses.
        let fake = Arc::new(
            FakeProvider::new()
                .err(
                    crate::retry::FailureClass::TruncatedEnvelope,
                    "EOF while parsing a value at line 317",
                )
                .ok(r#"{"score": 9}"#),
        );
        let result = extract_with(fake).await;
        assert_eq!(result.expect("recovery").score, 9);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn recovers_after_llm_parse_failure_via_reprompt() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(
            FakeProvider::new()
                .ok("this is not JSON at all")
                .ok(r#"{"score": 7}"#),
        );
        let result = extract_with(fake).await;
        assert_eq!(result.expect("recovery after re-prompt").score, 7);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn exhausts_attempts_with_bounded_call_count() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy()); // 3 attempts
        let fake = Arc::new(
            FakeProvider::new()
                .err(crate::retry::FailureClass::Transport, "always down")
                .err(crate::retry::FailureClass::Transport, "always down")
                .err(crate::retry::FailureClass::Transport, "always down"),
        );
        let result = extract_with(fake).await;
        let failure = result.expect_err("must exhaust");
        assert_eq!(failure.final_class, crate::retry::FailureClass::Transport);
        assert_eq!(failure.failures.len(), 3);
        assert_eq!(failure.last_raw, None, "no text ever produced");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn non_retryable_error_propagates_immediately() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(FakeProvider::new().err(
            crate::retry::FailureClass::NonRetryable,
            "insufficient balance",
        ));
        let result = extract_with(fake).await;
        let failure = result.expect_err("non-retryable must abort");
        assert_eq!(
            failure.final_class,
            crate::retry::FailureClass::NonRetryable
        );
        assert_eq!(failure.failures.len(), 1, "single call, no retries");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn transport_final_failure_clears_earlier_text_from_last_raw() {
        // Mixed sequence: an early attempt produces text (parse failure), then
        // later attempts die on transport. The ticket comment must NOT label the
        // earlier text as 'last attempt' — last_raw must be None because the
        // final attempt died before producing text.
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy()); // 3 attempts
        let fake = Arc::new(
            FakeProvider::new()
                .ok("this is not JSON")
                .err(crate::retry::FailureClass::Transport, "body read failed")
                .err(crate::retry::FailureClass::Transport, "body read failed"),
        );
        let result = extract_with(fake).await;
        let failure = result.expect_err("must exhaust on transport");
        assert_eq!(failure.final_class, crate::retry::FailureClass::Transport);
        assert_eq!(
            failure.last_raw, None,
            "earlier attempt text must not be presented as the last attempt"
        );
        assert_eq!(failure.failures.len(), 3);
        // The first failure is a parse (re-prompted), the last two transport.
        assert_eq!(failure.failures[0].class, crate::retry::FailureClass::Parse);
        assert_eq!(
            failure.failures[1].class,
            crate::retry::FailureClass::Transport
        );
        assert_eq!(
            failure.failures[2].class,
            crate::retry::FailureClass::Transport
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn tool_call_final_attempt_keeps_empty_last_raw() {
        // A tool-call final attempt (empty text) survives as Some("") so the
        // comment can state "final attempt was a tool call".
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy()); // 3 attempts
        let fake = Arc::new(
            FakeProvider::new().ok("not json").ok("not json").ok(""), // tool-call-style final attempt: no text
        );
        let result = extract_with(fake).await;
        let failure = result.expect_err("must exhaust");
        assert_eq!(failure.last_raw, Some(String::new()));
        assert_eq!(
            failure.failures.last().map(|r| r.class),
            Some(crate::retry::FailureClass::Parse)
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn attempts_byte_identical_except_reprompt() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(
            FakeProvider::new()
                .err(crate::retry::FailureClass::Transport, "transient")
                .ok("garbage text")
                .ok(r#"{"score": 6}"#),
        );
        let result = extract_with(fake.clone()).await;
        assert_eq!(result.expect("recovers").score, 6);

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(fingerprints.len(), 3);
        // Attempt 1 → 2: transport retry — byte-identical (no re-prompt).
        assert_eq!(
            fingerprints[0], fingerprints[1],
            "transport retry must be byte-identical"
        );
        // Attempt 2 → 3: parse failure — re-prompt appended (messages grew).
        assert_ne!(
            fingerprints[1], fingerprints[2],
            "parse-failure re-prompt must extend the request"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn score_out_of_range_never_passes_and_reprompts() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(
            FakeProvider::new()
                .ok(r#"{"score": 255}"#) // out of [0,10]
                .ok(r#"{"score": 5}"#),
        );
        let result = extract_with(fake).await;
        let verdict = result.expect("re-prompted after out-of-range score");
        assert_eq!(verdict.score, 5);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn score_out_of_range_all_attempts_classifies_as_out_of_range() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy()); // 3 attempts
        let fake = Arc::new(
            FakeProvider::new()
                .ok(r#"{"score": 101}"#) // just above the accepted [11,100] band
                .ok(r#"{"score": 101}"#)
                .ok(r#"{"score": 101}"#),
        );
        let result = extract_with(fake).await;
        let failure = result.expect_err("garbage score must never pass");
        assert_eq!(
            failure.final_class,
            crate::retry::FailureClass::OutOfRangeScore,
            "all-attempts-out-of-range must classify explicitly"
        );
        // Every failed attempt was an out-of-range-score rejection.
        assert!(
            failure
                .failures
                .iter()
                .all(|r| r.class == crate::retry::FailureClass::OutOfRangeScore)
        );
        assert_eq!(failure.failures.len(), 3);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn wall_clock_cap_binds_before_attempt_exhaustion() {
        let _guard = retry_tests_lock();
        // 7 attempts but a 100 ms wall cap and 1 s backoff — the cap binds.
        let _policy_guard =
            crate::util::test::install_test_retry_policy(crate::retry::RetryPolicy {
                max_attempts: 7,
                base_backoff_ms: 1_000,
                max_backoff_ms: 1_000,
                operation_timeout: std::time::Duration::from_millis(100),
                idle_timeout: std::time::Duration::from_secs(1),
            });
        let fake = Arc::new(
            FakeProvider::new()
                .err(crate::retry::FailureClass::Transport, "slow outage")
                .err(crate::retry::FailureClass::Transport, "slow outage"),
        );
        let result = extract_with(fake).await;
        let failure = result.expect_err("wall-clock cap must bind");
        assert_eq!(
            failure.final_class,
            crate::retry::FailureClass::WallClockExceeded
        );
        assert!(
            failure.failures.len() <= 2,
            "cap must stop the loop before 7 attempts"
        );
    }

    /// End-to-end regression guard for the retry-cause telemetry: recorded
    /// `llm_requests` rows carry the caller's base params (response_format 0
    /// — no wire forcing), and `retry_failures` rows carry the stamped
    /// purpose/role/workspace context plus the fine-grained JSON-quality
    /// cause. Catches the extraction path drifting from the recorded request
    /// or dropping the retry context.
    #[tokio::test]
    #[allow(clippy::too_many_lines)] // comprehensive end-to-end telemetry guard
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn extraction_rows_record_base_params_and_retry_context() {
        let _guard = retry_tests_lock();
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _store_guard = crate::util::test::install_test_log_store(store.clone());
        // Base params + meta set so the extraction path records rows
        // (meta-less requests are skipped by design).
        let params = ChatRequest {
            meta: Some(crate::ChatRequestMeta {
                purpose: "extraction",
                agent_id: "extraction-flag-test".to_string(),
                role: "reviewer".to_string(),
                workspace: "ws1".to_string(),
                ticket_id: None,
            }),
            ..test_params()
        };
        // Success path: recorded row carries the caller's response_format 0.
        let fake = Arc::new(FakeProvider::new().ok(r#"{"score": 8}"#));
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _fake_guard = install_fake_provider(provider);
        let result = retry_extract_structured_scoped::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &params,
            Some(&score_validator),
            None,
        )
        .await;
        assert_eq!(result.expect("extraction succeeds").score, 8);
        // Parse-failure path (re-prompted, recovers): the retry_failures row
        // carries the fine-grained cause + stamped context.
        let fake = Arc::new(FakeProvider::new().ok("garbage").ok(r#"{"score": 7}"#));
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _fake_guard = install_fake_provider(provider);
        let result = retry_extract_structured_scoped::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &params,
            Some(&score_validator),
            None,
        )
        .await;
        assert_eq!(result.expect("recovers after re-prompt").score, 7);
        // Failure path (all attempts die): rows keep the base params,
        // cost/provider stay NULL — no envelope to read them from.
        let fake = Arc::new(
            FakeProvider::new()
                .err(crate::retry::FailureClass::Transport, "always down")
                .err(crate::retry::FailureClass::Transport, "always down")
                .err(crate::retry::FailureClass::Transport, "always down"),
        );
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _fake_guard = install_fake_provider(provider);
        let result = retry_extract_structured_scoped::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &params,
            Some(&score_validator),
            None,
        )
        .await;
        assert!(
            result.is_err(),
            "all-attempt failure must not produce a verdict"
        );
        let rows = store
            .conn
            .query(
                "SELECT response_format, cost, upstream_provider, success \
                 FROM llm_requests WHERE agent_id = ?1 ORDER BY rowid",
                crate::turso::params!["extraction-flag-test"],
            )
            .await
            .expect("query recorded rows");
        let mut rows = rows.into_iter();
        let success = rows.next().expect("success row must exist");
        assert_eq!(success.get::<i64>(0).expect("response_format"), 0);
        assert_eq!(success.get::<i64>(3).expect("success"), 1);
        let recovery = rows.next().expect("recovery row must exist");
        assert_eq!(recovery.get::<i64>(0).expect("response_format"), 0);
        assert_eq!(recovery.get::<i64>(3).expect("success"), 1);
        let failure = rows.next().expect("failure row must exist");
        assert_eq!(failure.get::<i64>(0).expect("response_format"), 0);
        assert_eq!(failure.get::<i64>(3).expect("success"), 0);
        assert_eq!(
            failure.get::<Option<f64>>(1).expect("cost"),
            None,
            "no envelope on failure path — cost must stay NULL"
        );
        assert_eq!(
            failure.get::<Option<String>>(2).expect("upstream_provider"),
            None,
            "no envelope on failure path — provider must stay NULL"
        );
        // retry_failures rows: the parse failure carries cause=non_json, the
        // transport failures carry an empty cause — all with the stamped
        // operation context.
        let rf_rows = store
            .conn
            .query(
                "SELECT attempt, failure_class, purpose, role, workspace, cause \
                 FROM retry_failures WHERE role = 'reviewer' ORDER BY rowid",
                crate::turso::params![],
            )
            .await
            .expect("query retry failure rows");
        let rf: Vec<(i64, String, String, String, String, String)> = rf_rows
            .into_iter()
            .map(|r| {
                (
                    r.get::<i64>(0).expect("attempt"),
                    r.get::<String>(1).expect("class"),
                    r.get::<String>(2).expect("purpose"),
                    r.get::<String>(3).expect("role"),
                    r.get::<String>(4).expect("workspace"),
                    r.get::<String>(5).expect("cause"),
                )
            })
            .collect();
        assert_eq!(rf.len(), 4, "1 parse + 3 transport failures recorded");
        assert!(rf.iter().all(|(_, _, purpose, role, workspace, _)| {
            purpose == "extraction" && role == "reviewer" && workspace == "ws1"
        }));
        assert_eq!(rf[0].5, "non_json", "parse failure cause");
        assert!(rf[1..].iter().all(|(_, class, ..)| class == "transport"));
        assert!(
            rf[1..]
                .iter()
                .all(|(_, _, _, _, _, cause)| cause.is_empty())
        );
        assert!(rows.next().is_none(), "exactly three rows");
    }
}
