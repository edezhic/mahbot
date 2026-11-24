//! Structured extraction from conversation history.
//!
//! Provides LLM-powered extraction of structured data (JSON) from agent conversation history, with
//! retry logic.

use serde::de::DeserializeOwned;
use std::time::{Duration, Instant};

use crate::prompt::load_prompt;
use crate::retry::{
    FailureClass, RetryExhausted, RetryFailureRecord, RetryLoop, RetryPolicy, fail_exhausted,
};
use crate::util::json::parse_fenced_json;
use crate::{ChatMessage, ChatRequest, ExtractionValidator};

/// How one extraction departs from the shared schedule. [`Default`] is what every caller that does
/// not opt in gets, so a rule never changes an existing extraction.
#[derive(Clone, Copy)]
pub(crate) struct Rules {
    /// Re-ask with what was wrong with the previous reply, instead of the fixed sentence. Only the
    /// validator's own message can tell the model how its JSON missed.
    pub feedback: bool,
    /// A reply cut short by the output limit ends the attempts: re-asking the same request cannot
    /// produce a complete document.
    pub truncation_terminal: bool,
    /// The whole operation's wall-clock bound. Delivery waits on some extractions, so they cannot
    /// spend the ladder's idle timeouts.
    pub deadline: Option<Duration>,
    /// The class a validator rejection is recorded under: the caller's own schema conformance
    /// ([`FailureClass::SchemaMismatch`]) reads differently in `llm_failures` than a score outside
    /// its range.
    pub validator_class: FailureClass,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            feedback: false,
            truncation_terminal: false,
            deadline: None,
            validator_class: FailureClass::OutOfRangeScore,
        }
    }
}

/// The one record a spent deadline leaves. Its own class, so `llm_failures` and the stats rows never
/// read a spent budget as a provider failure.
fn deadline_record() -> RetryFailureRecord {
    RetryFailureRecord::with_metadata(
        FailureClass::Deadline,
        &anyhow::anyhow!("the extraction deadline was reached"),
        None,
        None,
    )
}

/// Scoped structured extraction with the hardened outer retry loop ([`crate::retry`] — default 13
/// attempts). Used by verdict extraction, diagnostics discovery, comment summaries, and research
/// orchestration. `policy_override` supplies a shorter schedule for fail-open comment-only callers
/// ([`crate::retry::RetryPolicy::comment`] — 3 attempts); `None` uses the default policy.
///
/// [`Rules::default()`] — the fixed re-ask, truncation retried, no wall-clock bound — is what every
/// existing caller gets; [`retry_extract_structured_with`] is the same loop for a caller that needs
/// its own rules.
pub(crate) async fn retry_extract_structured_scoped<T: DeserializeOwned>(
    history: &[ChatMessage],
    extraction_prompt: &str,
    params: &ChatRequest,
    validate: Option<&ExtractionValidator<T>>,
    policy_override: Option<&RetryPolicy>,
) -> Result<T, RetryExhausted> {
    retry_extract_structured_with(
        history,
        extraction_prompt,
        params,
        validate,
        policy_override,
        Rules::default(),
    )
    .await
}

/// [`retry_extract_structured_scoped`] with per-operation [`Rules`].
///
/// The outer loop is the SINGLE retry authority — provider-internal retries are suppressed via
/// [`crate::providers::chat_scoped`], so total provider HTTP calls per operation are bounded by
/// `max_attempts`.
///
/// Retry semantics per attempt:
/// - Provider/transport/truncation failure → retry byte-identical (unless
///   [`Rules::truncation_terminal`], where a reply cut by the output limit ends the attempts).
/// - Tool call, unparseable text, or validation failure (`validate` rejects)
///   → treat as parse failure: push raw text + `extraction/retry.md` re-prompt
///   (the ONLY permitted request mutation — extends the cached prefix); with [`Rules::feedback`]
///   the re-prompt is `extraction/schema_retry.md` naming what was wrong. An
///   empty last_raw pushes the `extraction/retry_no_output.md` marker instead
///   (a non-empty assistant turn is required to re-prompt).
/// - Valid JSON matching `T` and passing `validate` → return immediately.
///
/// `validate` (typically a score ∈ [0,10] check for [`crate::Verdict`]) runs fail-closed: a
/// rejected value is a parse failure, and if all attempts yield rejected values the operation fails
/// with an out-of-range-score classification — a garbage score never passes any gate. It must be
/// `Send + Sync` because extraction runs inside `join_all` futures.
///
/// Terminal failure is [`RetryExhausted`] — it carries the last-attempt raw text (`last_raw`, for
/// verdict-extraction ticket comments) plus the per-attempt diagnostics trail.
///
/// The history is bounded by the request picture cap before the first attempt, so every retry sends
/// the same bounded request.
#[expect(clippy::too_many_lines)] // one loop: attempt, classify, re-ask
pub(crate) async fn retry_extract_structured_with<T: DeserializeOwned>(
    history: &[ChatMessage],
    extraction_prompt: &str,
    params: &ChatRequest,
    validate: Option<&ExtractionValidator<T>>,
    policy_override: Option<&RetryPolicy>,
    rules: Rules,
) -> Result<T, RetryExhausted> {
    let mut extraction_history = history.to_vec();
    let operation_started = Instant::now();

    // Only push the extraction prompt if non-empty — caller may have embedded it
    if !extraction_prompt.is_empty() {
        extraction_history.push(ChatMessage::user(extraction_prompt));
    }
    // No session behind this history, so `None` counts it. Bounded once here, so every retry below
    // sends the same bounded request.
    crate::session::image_cap::apply(&mut extraction_history, None);

    // Telemetry record request: the recorded request derives from params so recorded metadata can
    // never diverge from the bytes actually sent. Only the telemetry fields are read from it —
    // messages are emptied here so the per-attempt spread doesn't clone the caller's history; the
    // one-time params.clone() below is that cost.
    let mut record_request = ChatRequest {
        messages: vec![],
        ..params.clone()
    };

    let retry_prompt = load_prompt("extraction/retry.md");
    let policy = policy_override
        .cloned()
        .unwrap_or_else(RetryPolicy::current);
    let mut loop_state = RetryLoop::new_scoped(&policy, &record_request);
    let mut last_raw: Option<String> = None;

    for attempt in 1..=policy.max_attempts {
        // The whole operation is bounded when delivery waits on it: a spent budget starts no
        // attempt at all (a timeout alone would still let an immediately-ready call through), and
        // the remaining time caps the attempt that does start, so a stalled provider cannot hold
        // the answer past the deadline.
        let remaining = rules
            .deadline
            .map(|limit| limit.saturating_sub(operation_started.elapsed()));
        if remaining.is_some_and(|left| left.is_zero()) {
            loop_state.record(deadline_record());
            break;
        }
        loop_state.begin_attempt(Instant::now());
        let request = ChatRequest {
            messages: extraction_history.clone(),
            ..record_request.clone()
        };

        let call = crate::providers::chat_scoped(request);
        let outcome = match remaining {
            Some(left) => {
                let Ok(outcome) = tokio::time::timeout(left, call).await else {
                    loop_state.record(deadline_record());
                    break;
                };
                outcome
            }
            None => call.await,
        };
        match outcome {
            Ok(response) => {
                last_raw = Some(response.text_or_empty().to_string());

                // A reply cut short by the output limit cannot be re-asked into a complete
                // document, so it ends the attempts instead of spending the budget.
                if rules.truncation_terminal && response.finish_reason.as_deref() == Some("length")
                {
                    let err = anyhow::anyhow!("the reply was cut short by the output limit");
                    loop_state.record(RetryFailureRecord::with_metadata(
                        FailureClass::TruncatedOutput,
                        &err,
                        response.finish_reason.clone(),
                        None,
                    ));
                    break;
                }

                let (class, detail, feedback): (FailureClass, String, String) =
                    if response.tool_calls.is_empty() {
                        let raw = last_raw.as_deref().unwrap_or_default();
                        match parse_fenced_json::<T>(raw) {
                            Ok(result) => {
                                if let Some(validate) = validate
                                    && let Err(msg) = validate(&result)
                                {
                                    (
                                        rules.validator_class,
                                        format!("extracted value rejected by validation: {msg}"),
                                        msg,
                                    )
                                } else {
                                    crate::stats::record_llm_success(
                                        &record_request,
                                        operation_started,
                                        attempt,
                                        &response,
                                    );
                                    return Ok(result);
                                }
                            }
                            Err(e) => (
                                FailureClass::Parse,
                                format!("failed to parse extracted JSON: {e}"),
                                "the reply was not valid JSON".to_string(),
                            ),
                        }
                    } else {
                        (
                            FailureClass::Parse,
                            "extraction attempt returned a tool call instead of JSON".to_string(),
                            "you called a tool instead of replying".to_string(),
                        )
                    };

                let err = anyhow::anyhow!("{detail}");
                let rec = RetryFailureRecord::with_metadata(
                    class,
                    &err,
                    response.finish_reason.clone(),
                    None,
                );
                loop_state.record(rec);

                // Re-prompt: push raw assistant text + retry prompt. Only needed when another
                // attempt will actually run — the final attempt's re-prompt is never sent.
                if attempt < policy.max_attempts {
                    // An empty assistant turn gives the model nothing to correct — replay a
                    // non-empty marker instead.
                    let prev = last_raw.clone().filter(|text| !text.trim().is_empty());
                    extraction_history.push(ChatMessage::assistant(
                        prev.unwrap_or_else(|| load_prompt("extraction/retry_no_output.md")),
                    ));
                    let reask = if rules.feedback {
                        crate::prompt::substitute(
                            &load_prompt("extraction/schema_retry.md"),
                            &[("{{error}}", &feedback)],
                        )
                    } else {
                        retry_prompt.clone()
                    };
                    extraction_history.push(ChatMessage::user(reask.as_str()));
                }
            }

            Err(scoped_err) => {
                // The final attempt died before producing text (transport / truncation / budget) —
                // clear any earlier text so the ticket comment never labels it 'last attempt'.
                last_raw = None;
                let non_retryable = !scoped_err.class.is_retryable();
                loop_state.record(scoped_err.record);
                if non_retryable {
                    let exhausted = RetryExhausted::with_last_raw(
                        loop_state.into_failures(),
                        scoped_err.class,
                        last_raw,
                    );
                    return fail_exhausted(&record_request, operation_started, exhausted);
                }
                // A dead pin does not get the rest of the ladder (see `providers::selection`).
                loop_state.retarget_after_failure(scoped_err.class, &mut record_request);
            }
        }

        // The sleep is part of the same bound: delivery waits on the whole operation, so a backoff
        // that would run past the deadline ends the attempts instead of being served.
        let slept = match rules
            .deadline
            .map(|limit| limit.saturating_sub(operation_started.elapsed()))
        {
            Some(left) => tokio::time::timeout(left, loop_state.sleep_between(attempt)).await,
            None => Ok(loop_state.sleep_between(attempt).await),
        };
        match slept {
            Ok(Err(class)) => {
                let exhausted =
                    RetryExhausted::with_last_raw(loop_state.into_failures(), class, last_raw);
                return fail_exhausted(&record_request, operation_started, exhausted);
            }
            Ok(Ok(())) => {}
            Err(_) => {
                loop_state.record(deadline_record());
                break;
            }
        }
    }

    // Exhausted — report the last failure's class so operators see e.g. out-of-range-score (never a
    // garbage pass) rather than a generic class.
    let final_class = loop_state.final_class();
    let exhausted =
        RetryExhausted::with_last_raw(loop_state.into_failures(), final_class, last_raw);
    fail_exhausted(&record_request, operation_started, exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::tiny_test_policy;
    use crate::util::test::{FakeProvider, install_fake_provider};
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
            max_tokens: None,
            reasoning_effort: None,
            provider: None,
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

    /// The same extraction under a caller's own [`Rules`].
    async fn extract_with_rules(
        fake: Arc<FakeProvider>,
        rules: Rules,
    ) -> Result<FakeVerdict, crate::retry::RetryExhausted> {
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _guard = install_fake_provider(provider);
        retry_extract_structured_with::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &test_params(),
            Some(&score_validator),
            None,
            rules,
        )
        .await
    }

    /// A reply cut short by the output limit ends the attempts under the schema's own rules: the same
    /// request re-asked cannot produce a complete document, so the budget is not spent on it.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn a_truncated_reply_ends_the_attempts_at_once() {
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy()); // 3 attempts
        let fake = Arc::new(
            FakeProvider::new()
                .ok_with_finish("{\"score\": 1", Some("length"))
                .ok(r#"{"score": 5}"#),
        );
        let failure = extract_with_rules(
            fake.clone(),
            Rules {
                truncation_terminal: true,
                ..Rules::default()
            },
        )
        .await
        .expect_err("a truncated reply cannot be re-asked into a complete one");
        assert_eq!(
            failure.final_class,
            crate::retry::FailureClass::TruncatedOutput,
            "a reply cut by the output limit classifies as truncated"
        );
        assert_eq!(failure.failures.len(), 1, "one attempt, no retry");
        assert_eq!(
            fake.request_fingerprints.lock().unwrap().len(),
            1,
            "one provider call"
        );
    }

    /// The deadline bounds the BACKOFF too: a ladder that would sleep past it ends the attempts
    /// instead of holding delivery, so the whole operation really is bounded.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn the_deadline_bounds_the_backoff() {
        let fake = Arc::new(FakeProvider::new().ok("not json"));
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _guard = install_fake_provider(provider);
        let started = std::time::Instant::now();
        let failure = retry_extract_structured_with::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &test_params(),
            Some(&score_validator),
            Some(&crate::retry::RetryPolicy {
                max_attempts: 3,
                base_backoff_ms: 60_000,
                max_backoff_ms: 60_000,
            }),
            Rules {
                deadline: Some(Duration::from_millis(50)),
                ..Rules::default()
            },
        )
        .await
        .expect_err("the backoff would run past the deadline");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the deadline bounds the backoff: {:?}",
            started.elapsed()
        );
        assert_eq!(
            fake.request_fingerprints.lock().unwrap().len(),
            1,
            "no second attempt after the deadline"
        );
        assert_eq!(
            failure.failures.len(),
            2,
            "the parse failure, then the deadline that ended the backoff"
        );
        assert_eq!(failure.final_class, crate::retry::FailureClass::Deadline);
    }

    /// With feedback, the re-ask carries the validator's own message: only that can tell the model
    /// how its JSON missed.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn a_rejected_reply_is_re_asked_with_what_was_wrong() {
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(
            FakeProvider::new()
                .ok(r#"{"score": 101}"#) // out of [0,10]
                .ok(r#"{"score": 5}"#),
        );
        let verdict = extract_with_rules(
            fake.clone(),
            Rules {
                feedback: true,
                ..Rules::default()
            },
        )
        .await
        .expect("the second reply passes");
        assert_eq!(verdict.score, 5);

        let messages = fake.request_messages.lock().unwrap().clone();
        assert_eq!(messages.len(), 2);
        assert!(
            messages[1].contains("score 101 out of range"),
            "the validator's own message is the re-ask: {}",
            messages[1]
        );
        assert!(
            !messages[1].contains("not valid JSON"),
            "the fixed sentence is replaced: {}",
            messages[1]
        );
    }

    /// The whole operation is bounded by its deadline: a stalled provider ends the attempts within
    /// the bound instead of holding the caller for the ladder's idle timeouts.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn the_deadline_bounds_the_whole_operation() {
        /// A provider that never answers, so only the deadline can end the attempt.
        struct Stalled;

        #[async_trait::async_trait]
        impl crate::Provider for Stalled {
            async fn chat_scoped(
                &self,
                _request: ChatRequest,
            ) -> Result<crate::ChatResponse, crate::providers::ScopedCallError> {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Ok(crate::ChatResponse::default())
            }

            async fn warmup(&self) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let provider: Arc<dyn crate::Provider> = Arc::new(Stalled);
        let _guard = install_fake_provider(provider);
        let started = std::time::Instant::now();
        let failure = retry_extract_structured_with::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &test_params(),
            Some(&score_validator),
            None,
            Rules {
                deadline: Some(std::time::Duration::from_millis(50)),
                ..Rules::default()
            },
        )
        .await
        .expect_err("the stalled attempt never answers");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the deadline bounds the operation: {:?}",
            started.elapsed()
        );
        assert_eq!(failure.failures.len(), 1, "the deadline is not retried");
        assert_eq!(failure.final_class, crate::retry::FailureClass::Deadline);
    }

    /// The default rules keep the fixed re-ask sentence every existing extraction was built on.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn the_default_rules_re_ask_with_the_fixed_sentence() {
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(FakeProvider::new().ok("garbage text").ok(r#"{"score": 5}"#));
        let verdict = extract_with(fake.clone())
            .await
            .expect("recovers after a re-ask");
        assert_eq!(verdict.score, 5);
        let messages = fake.request_messages.lock().unwrap().clone();
        assert_eq!(messages.len(), 2);
        assert!(
            messages[1].contains(load_prompt("extraction/retry.md").trim()),
            "the fixed sentence is the re-ask: {}",
            messages[1]
        );
    }

    /// A provider that has been failing for the whole window does not get the rest of the ladder:
    /// the remaining attempts go to the next entry of the model's priority list. Without this the
    /// run would fail at extraction with a live provider one entry down.
    #[tokio::test]
    #[serial_test::serial(config_persist, provider)] // the routing row, the fake provider and the outage map are process-global
    async fn a_dead_provider_is_left_mid_ladder() {
        use crate::providers::selection;

        crate::util::test::init_test_stores().await;
        let model = "extraction/provider-switch";
        // Guarded: the priority list lives in the process-global CONFIG, and a panic mid-test must
        // not leave these providers installed for the rest of the binary.
        let _order = crate::config::test_routing_order(model, "first, second");
        selection::force_outage(
            model,
            "first",
            std::time::Instant::now()
                .checked_sub(selection::MOVE_AFTER + std::time::Duration::from_secs(1))
                .expect("a five-minute subtraction stays representable"),
        );

        let fake = Arc::new(
            FakeProvider::new()
                .err(
                    crate::retry::FailureClass::Transport,
                    "the provider is down",
                )
                .ok(r#"{"score": 8}"#),
        );
        let provider: Arc<dyn crate::Provider> = fake.clone();
        let _guard = install_fake_provider(provider);
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());

        let mut params = test_params();
        params.model = model.to_string();
        params.provider = Some("first".to_string());
        let verdict = retry_extract_structured_scoped::<FakeVerdict>(
            &history(),
            "return JSON verdict",
            &params,
            Some(&score_validator),
            None,
        )
        .await
        .expect("the second attempt is served");
        assert_eq!(verdict.score, 8);

        let fingerprints = fake.request_fingerprints.lock().unwrap().clone();
        assert_eq!(fingerprints.len(), 2);
        assert!(
            fingerprints[0].contains(r#"provider: Some("first")"#),
            "the first attempt stays on the pinned provider: {}",
            fingerprints[0]
        );
        assert!(
            fingerprints[1].contains(r#"provider: Some("second")"#),
            "the attempt after the window goes to the next available entry: {}",
            fingerprints[1]
        );
    }

    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn recovers_after_retryable_failures() {
        let cases = [
            (
                "transport errors",
                FakeProvider::new()
                    .err(crate::retry::FailureClass::Transport, "transport error 1")
                    .err(crate::retry::FailureClass::Transport, "transport error 2")
                    .ok(r#"{"score": 8}"#),
                8,
            ),
            (
                "truncated envelope",
                FakeProvider::new()
                    .err(
                        crate::retry::FailureClass::TruncatedEnvelope,
                        "EOF while parsing a value at line 317",
                    )
                    .ok(r#"{"score": 9}"#),
                9,
            ),
        ];
        for (reason, fake, score) in cases {
            let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
            let verdict = extract_with(Arc::new(fake))
                .await
                .expect("should recover after retryable failures");
            assert_eq!(verdict.score, score, "case: {reason}");
        }
    }

    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn exhausts_attempts_with_bounded_call_count() {
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
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn non_retryable_error_propagates_immediately() {
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let fake = Arc::new(FakeProvider::new().err(
            crate::retry::FailureClass::InvalidRequest,
            "insufficient balance",
        ));
        let result = extract_with(fake).await;
        let failure = result.expect_err("non-retryable must abort");
        assert_eq!(
            failure.final_class,
            crate::retry::FailureClass::InvalidRequest
        );
        assert_eq!(failure.failures.len(), 1, "single call, no retries");
    }

    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn transport_final_failure_clears_earlier_text_from_last_raw() {
        // Mixed sequence: an early attempt produces text (parse failure), then later attempts die
        // on transport. The ticket comment must NOT label the earlier text as 'last attempt' —
        // last_raw must be None because the final attempt died before producing text.
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
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn attempts_byte_identical_except_reprompt() {
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
        assert_eq!(
            fingerprints[0], fingerprints[1],
            "transport retry must be byte-identical"
        );
        assert_ne!(
            fingerprints[1], fingerprints[2],
            "parse-failure re-prompt must extend the request"
        );
    }

    /// The parse-failure re-prompt turns the failed reply into the appended assistant turn: a
    /// JSON-shaped reply is carried verbatim (the production trigger), and a reply with NO text — a
    /// tool-call-style response — is replaced by the non-empty no-output marker rather than
    /// replayed as an empty frame.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn parse_failure_reprompts_with_a_non_empty_assistant_turn() {
        let cases = [
            (
                "a JSON reply is carried verbatim",
                r#"{"verdict": "bad"}"#,
                r#"{"score": 4}"#,
                4u8,
                Some(r#"{\"verdict\": \"bad\"}"#.to_string()),
                None,
            ),
            (
                "a reply with no text becomes the non-empty marker",
                "",
                r#"{"score": 7}"#,
                7u8,
                None,
                Some("content: \"\"".to_string()),
            ),
        ];
        for (reason, first, second, score, must_contain, must_not_contain) in cases {
            let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
            let fake = Arc::new(FakeProvider::new().ok(first).ok(second));
            let result = extract_with(fake.clone()).await;
            assert_eq!(result.expect("recovers").score, score, "case: {reason}");

            // This seam verifies ChatMessages; the wire form is pinned by the decode-gate and
            // serializer tests.
            let messages = fake.request_messages.lock().unwrap().clone();
            assert_eq!(messages.len(), 2, "case: {reason}");
            assert!(messages[1].contains("Assistant"), "{}", messages[1]);
            if let Some(expected) = &must_contain {
                assert!(
                    messages[1].contains(expected.as_str()),
                    "case: {reason}: {}",
                    messages[1]
                );
            }
            if let Some(never) = must_not_contain {
                assert!(
                    !messages[1].contains(&never),
                    "case: {reason}: {}",
                    messages[1]
                );
            }
        }
    }

    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn score_out_of_range_never_passes_and_reprompts() {
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
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn score_out_of_range_all_attempts_classifies_as_out_of_range() {
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
        assert!(
            failure
                .failures
                .iter()
                .all(|r| r.class == crate::retry::FailureClass::OutOfRangeScore)
        );
        assert_eq!(failure.failures.len(), 3);
    }

    /// End-to-end regression guard for the retry telemetry: recorded `llm_requests` rows carry the
    /// caller's base params, and the failure row keeps cost/provider NULL (no envelope on the
    /// failure path). Catches the extraction path drifting from the recorded request.
    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn extraction_rows_record_base_params_and_retry_context() {
        let _policy_guard = crate::util::test::install_test_retry_policy(tiny_test_policy());
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _store_guard = crate::util::test::install_test_log_store(store.clone());
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
        crate::telemetry::drain_for_test().await;
        let rows = store
            .conn
            .query(
                "SELECT cost, upstream_provider, success \
                 FROM llm_requests WHERE agent_id = ?1 ORDER BY rowid",
                crate::db::params!["extraction-flag-test"],
            )
            .await
            .expect("query recorded rows");
        let mut rows = rows.into_iter();
        let success = rows.next().expect("success row must exist");
        assert_eq!(success.get::<i64>(2).expect("success"), 1);
        let recovery = rows.next().expect("recovery row must exist");
        assert_eq!(recovery.get::<i64>(2).expect("success"), 1);
        let failure = rows.next().expect("failure row must exist");
        assert_eq!(failure.get::<i64>(2).expect("success"), 0);
        assert_eq!(
            failure.get::<Option<f64>>(0).expect("cost"),
            None,
            "no envelope on failure path — cost must stay NULL"
        );
        assert_eq!(
            failure.get::<Option<String>>(1).expect("upstream_provider"),
            None,
            "no envelope on failure path — provider must stay NULL"
        );
        assert!(rows.next().is_none(), "exactly three rows");
    }
}
