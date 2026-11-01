//! Outer retry orchestration for LLM operations.
//!
//! The expensive LLM paths retry through this module — agent-loop chat calls, structured extraction
//! (verdicts, diagnostics discovery, comment summaries, research orchestration), grouping repair
//! and deep-research synthesis. The list is intentionally non-exhaustive: behaviour is defined by
//! the policy constructors, not by per-path documentation.
//!
//! This outer loop is the **single retry authority**: on these calls provider-internal retries are
//! suppressed (see [`Provider::chat_scoped`]), so total provider HTTP calls per operation are
//! bounded by the policy in effect.
//!
//! All request parameters (model, messages, tools, max_tokens, reasoning_effort, provider routing)
//! are byte-identical across ALL attempts — reasoning_effort is FIXED (prefix cache preservation,
//! never lower it) and no escalation of any kind. The only permitted change is appended feedback on
//! a failed attempt (the extraction re-prompt, repair-round instructions, synthesis feedback),
//! which extends the cached prefix.
//!
//! Four policies, snapshotted once at operation start ([`RetryPolicy`]). All share the same
//! mechanics: ±25% jitter on the SLEEP ONLY (never on request bytes); Retry-After honoured, clamped
//! [5000 ms, 60000 ms]; shutdown-abortable. They differ only in attempt counts and backoff — no
//! per-operation wall-clock cap exists; the only timing bound is the 600 s idle timeout
//! ([`DEFAULT_IDLE_TIMEOUT`]), so worst case is roughly `attempts × 600 s + backoff` (a body that
//! keeps trickling bytes is never cut):
//!
//! - Default ([`RetryPolicy::current`]) — 13 attempts over a 5/10/20/40/60/60… s ladder (base
//!   5000 ms, doubled to a 60000 ms cap; 555 s of sleep in total) to ride out a sustained 503.
//! - Synthesis ([`RetryPolicy::synthesis`]) — 3 total attempts (1 synthesis + up to 2 repair
//!   rounds), 30–45 s band, so a bad grouping degrades to the deterministic fallback after at most
//!   3 calls.
//! - Comment ([`RetryPolicy::comment`]) — 3 attempts on the default backoff, because a long retry
//!   would stall the pipeline for a non-critical extraction.
//! - Continuation ([`RetryPolicy::continuation`]) — 3 attempts with no inter-attempt sleep, the
//!   reasoning-only-stop recovery budget (see
//!   [`crate::agent::Agent::recover_reasoning_only_stop`]).
//!
//! Per-attempt timeout semantics come from [`Provider::chat_scoped`]: the header wait (TTFB) and
//! each body-read chunk wait are bounded by the 600 s idle timeout, resetting while data flows — a
//! healthy-but-slow generation is never cut, a stalled attempt is aborted and retried.
//!
//! Every failed attempt produces a [`RetryFailureRecord`] appended to the in-memory failure trail;
//! terminal exhaustion surfaces it through [`RetryExhausted`], which feeds the live `llm_requests`
//! stats rows. Every failed attempt of a metadata-carrying operation also persists one
//! `llm_failures` row, grouped by a per-operation id — see [`RetryLoop::record`].

use std::fmt;
use std::time::{Duration, Instant};

#[cfg(test)]
use crate::util::UnwrapPoison;
use crate::{ChatRequest, ChatResponse};

pub(crate) const DEFAULT_RETRY_MAX_ATTEMPTS: u32 = 13;
const DEFAULT_RETRY_BASE_BACKOFF_MS: u64 = 5_000;
const DEFAULT_RETRY_MAX_BACKOFF_MS: u64 = 60_000;

/// Joint-verdict synthesis schedule: total calls (1 synthesis + up to N-1 repair rounds; default 3,
/// the lower edge of the approved 3–5 band), 30–45 s backoff band (±25% jitter on sleeps).
const DEFAULT_SYNTHESIS_MAX_ATTEMPTS: u32 = 3;
const DEFAULT_SYNTHESIS_BASE_BACKOFF_MS: u64 = 30_000;
const DEFAULT_SYNTHESIS_MAX_BACKOFF_MS: u64 = 45_000;

/// Fail-open comment-extraction attempt budget — see [`RetryPolicy::comment`].
const DEFAULT_COMMENT_MAX_ATTEMPTS: u32 = 3;

/// Reasoning-only-stop continuation schedule: up to 3 appended-only re-requests after the original
/// in-class response, so a stuck reasoning-only model fails the turn safely. Each attempt is a
/// single `chat_scoped` call (no inner transport retry — the appended tail makes every new
/// reasoning state a fresh request; retryable transport errors re-send the identical bytes).
const DEFAULT_CONTINUATION_MAX_ATTEMPTS: u32 = 3;

/// The single timeout governing every LLM chat request — the response-header wait (TTFB) and each
/// body-read chunk wait, resetting while data flows. A stalled attempt is aborted at 600 s and
/// retried; a healthy-but-slow generation is never cut. No total request timeout applies.
pub(crate) const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Lower clamp for Retry-After honouring.
const RETRY_AFTER_MIN_MS: u64 = 5_000;
/// Upper clamp for Retry-After honouring — 60 s across all scoped retry paths.
pub(crate) const RETRY_AFTER_MAX_MS: u64 = 60_000;

// Retry tunables snapshotted at operation start. Hot-reload mid-retry must not change the schedule
// — callers snapshot once and reuse. Invalid configured values fall back to defaults.
#[derive(Debug, Clone)]
pub(crate) struct RetryPolicy {
    /// Total loop attempts (default 13). The operation makes at most this many provider HTTP calls.
    pub max_attempts: u32,
    /// Base backoff in milliseconds (default 5000).
    pub base_backoff_ms: u64,
    /// Backoff cap in milliseconds (default 60000).
    pub max_backoff_ms: u64,
}

impl RetryPolicy {
    #[must_use]
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_RETRY_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_RETRY_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_RETRY_MAX_BACKOFF_MS,
        }
    }

    #[must_use]
    pub(crate) fn current() -> Self {
        Self::default().apply_test_override()
    }

    /// The joint-verdict synthesis policy. `synthesis_max_attempts` is the TOTAL call count: 1
    /// synthesis + up to N-1 repair rounds. Bounded so a bad grouping pass degrades to the
    /// deterministic fallback comment after at most 3 calls.
    #[must_use]
    pub(crate) fn synthesis() -> Self {
        Self {
            max_attempts: DEFAULT_SYNTHESIS_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_SYNTHESIS_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_SYNTHESIS_MAX_BACKOFF_MS,
        }
        .apply_test_override()
    }

    /// The comment-only extraction policy for fail-open callers: the caller falls back to raw text
    /// on failure, so a bounded attempt count avoids stalling the pipeline for a non-critical
    /// operation (the 13-attempt budget is for verdict gates).
    #[must_use]
    pub(crate) fn comment() -> Self {
        Self {
            max_attempts: DEFAULT_COMMENT_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_RETRY_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_RETRY_MAX_BACKOFF_MS,
        }
        .apply_test_override()
    }

    /// The reasoning-only-stop continuation policy: bounded recovery for an empty-content/no-tool
    /// response. Mirrors [`Self::comment`]'s 3 attempts. No inter-attempt sleep — the appended-only
    /// tail makes each new reasoning state a fresh request with a byte-stable prefix (retryable
    /// transport errors re-send the identical bytes; non-retryable ones break immediately). The
    /// backoff fields are unused here.
    #[must_use]
    pub(crate) fn continuation() -> Self {
        Self {
            max_attempts: DEFAULT_CONTINUATION_MAX_ATTEMPTS,
            base_backoff_ms: 0,
            max_backoff_ms: 0,
        }
        .apply_test_override()
    }

    /// Test seam: in tests the override installed via [`swap_test_retry_policy`] takes precedence
    /// over the hardcoded schedule, so retry-loop tests don't sleep for minutes. Poison-tolerant: a
    /// failing test must not cascade. A no-op outside tests.
    #[must_use]
    fn apply_test_override(self) -> Self {
        #[cfg(test)]
        if let Some(p) = test_override() {
            return p;
        }
        self
    }
}

#[cfg(test)]
static TEST_POLICY_OVERRIDE: std::sync::RwLock<Option<RetryPolicy>> = std::sync::RwLock::new(None);

/// Read the installed test policy override, if any. Poison-tolerant.
#[cfg(test)]
fn test_override() -> Option<RetryPolicy> {
    let guard = TEST_POLICY_OVERRIDE.read().unwrap_poison();
    guard.as_ref().cloned()
}

#[cfg(test)]
pub(crate) fn swap_test_retry_policy(policy: RetryPolicy) -> Option<RetryPolicy> {
    let mut guard = TEST_POLICY_OVERRIDE.write().unwrap_poison();
    let previous = guard.take();
    *guard = Some(policy);
    previous
}

/// Restore a previously swapped-out test retry-policy override.
#[cfg(test)]
pub(crate) fn restore_test_retry_policy(previous: Option<RetryPolicy>) {
    *TEST_POLICY_OVERRIDE.write().unwrap_poison() = previous;
}

/// A tiny retry policy for tests: 3 attempts, ~1 ms backoff.
#[cfg(test)]
pub(crate) fn tiny_test_policy() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 3,
        base_backoff_ms: 1,
        max_backoff_ms: 1,
    }
}

/// The canonical backoff sleep sequence (ms) for a policy: pure doubling from `base_backoff_ms`,
/// capped at `max_backoff_ms`. The defaults (5000/60000) yield the agent-loop schedule
/// 5/10/20/40/60/60… s — total sleep 555 s over 13 attempts.
#[must_use]
fn backoff_sequence(policy: &RetryPolicy) -> Vec<u64> {
    let sleeps = policy.max_attempts.saturating_sub(1) as usize;
    let mut seq = Vec::with_capacity(sleeps);
    for i in 0..sleeps {
        let doubled = policy.base_backoff_ms.saturating_mul(1u64 << i.min(6));
        seq.push(doubled.min(policy.max_backoff_ms));
    }
    seq
}

/// ±25% jitter around `base_ms`: random within [75%, 125%) so parallel agents retrying the same
/// transient error don't synchronize. Jitter touches the SLEEP ONLY, never the request bytes. The
/// modulo guard keeps 0/1 ms test schedules division-safe.
#[must_use]
fn jittered_backoff_ms(base_ms: u64) -> u64 {
    base_ms - base_ms / 4 + (rand::random::<u64>() % (base_ms / 2).max(1))
}

/// The actual sleep for one inter-attempt gap. Retry-After (when present) is authoritative and
/// followed precisely, clamped to [5000 ms, 60000 ms]; otherwise ±25% jitter on the schedule
/// backoff.
#[must_use]
fn compute_sleep_ms(schedule_ms: u64, retry_after_ms: Option<u64>) -> u64 {
    if let Some(ra) = retry_after_ms {
        ra.clamp(RETRY_AFTER_MIN_MS, RETRY_AFTER_MAX_MS)
    } else {
        jittered_backoff_ms(schedule_ms)
    }
}

// Granular failure classification for the retry/error trail. `Membership`/`Completeness`/
// `ContradictionAgents`/`ValidationOther` carry repair-validation semantics on the shared
// provider-failure path: retryable, and only ever produced by the grouping core. `TruncatedOutput`
// marks an output-token-limit truncation (finish_reason "length") — retryable with shorten
// feedback.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailureClass {
    /// Network/transport error (connection reset, timeouts, 5xx, 429).
    Transport,
    /// Server truncated the response envelope: body-read error or content-length mismatch (the
    /// "error reading response body" / "EOF while parsing" defect class).
    TruncatedEnvelope,
    /// Body read fully but JSON did not parse (LLM format issue).
    Parse,
    /// Parsed, but a validation hook rejected the value (e.g. verdict score outside [0,10]).
    OutOfRangeScore,
    /// Repair-round group rejected: a member's item id is out of range, or the member was already
    /// placed in a frozen group / pinned by an accepted contradiction reference.
    Membership,
    /// Repair-round proposal silently dropped unfrozen items (incomplete coverage).
    Completeness,
    /// A proposed contradiction/reference lacks ≥2 distinct cited agents.
    ContradictionAgents,
    /// A validation rejection outside the granular categories (empty heading/summary, malformed
    /// structure, out-of-range group reference).
    ValidationOther,
    /// Provider returned text cut at its output-token limit (finish_reason "length").
    TruncatedOutput,
    /// Provider returned an empty text response.
    NoResponse,
    /// Permanent client error (auth, quota, invalid model, tool schema).
    NonRetryable,
    /// Global shutdown fired mid-operation.
    Shutdown,
}

impl FailureClass {
    #[must_use]
    pub(crate) const fn is_retryable(self) -> bool {
        !matches!(self, Self::NonRetryable | Self::Shutdown)
    }

    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::TruncatedEnvelope => "truncated_envelope",
            Self::Parse => "parse",
            Self::OutOfRangeScore => "out_of_range_score",
            Self::Membership => "membership",
            Self::Completeness => "completeness",
            Self::ContradictionAgents => "contradiction_agents",
            Self::ValidationOther => "validation_other",
            Self::TruncatedOutput => "truncated_output",
            Self::NoResponse => "no_response",
            Self::NonRetryable => "non_retryable",
            Self::Shutdown => "shutdown",
        }
    }
}

// Per-attempt failure diagnostics, carried in the in-memory retry trail.

#[derive(Debug, Clone)]
pub(crate) struct RetryFailureRecord {
    /// Granular failure classification.
    pub class: FailureClass,
    /// Full error cause chain.
    pub error_chain: String,
    /// `choices[0].finish_reason` parsed from the envelope (best effort).
    pub finish_reason: Option<String>,
    /// Retry-After value from the response, if any.
    pub retry_after_ms: Option<u64>,
}

impl RetryFailureRecord {
    /// Build a record from a plain error (no envelope telemetry available).
    #[must_use]
    pub(crate) fn new_simple(
        class: FailureClass,
        error: &anyhow::Error,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self {
            class,
            error_chain: format!("{error:#}"),
            finish_reason: None,
            retry_after_ms,
        }
    }

    /// Build a record with envelope telemetry (finish_reason).
    #[must_use]
    pub(crate) fn with_metadata(
        class: FailureClass,
        error: &anyhow::Error,
        finish_reason: Option<String>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self {
            finish_reason,
            ..Self::new_simple(class, error, retry_after_ms)
        }
    }
}

// Terminal failure of an outer retry loop: the per-attempt diagnostics trail plus the final
// classification, so callers can fail open with a useful reason (consolidation) or write the last
// raw text into a ticket comment (verdict extraction). `last_raw` is `Some(text)` when the last
// COMPLETED attempt produced assistant text (`Some("")` for a tool-call final attempt) and `None`
// when that attempt died before producing text — text from earlier attempts is never presented as
// the last one.
#[derive(Debug, Clone)]
pub(crate) struct RetryExhausted {
    pub failures: Vec<RetryFailureRecord>,
    pub final_class: FailureClass,
    /// Last-attempt raw text (verdict-extraction ticket-comment dumps). See the type docs for the
    /// precise "last completed attempt" semantics.
    pub last_raw: Option<String>,
    /// Human-readable terminal reason (identical to `Display`).
    pub detail: String,
}

impl RetryExhausted {
    #[must_use]
    fn with_trail(
        failures: Vec<RetryFailureRecord>,
        final_class: FailureClass,
        last_raw: Option<String>,
    ) -> Self {
        let detail = if let Some(last) = failures.last() {
            format!(
                "{} attempt(s) failed (last: {}): {}",
                failures.len(),
                final_class.label(),
                last.error_chain,
            )
        } else {
            format!("operation failed: {}", final_class.label())
        };
        Self {
            failures,
            final_class,
            last_raw,
            detail,
        }
    }

    #[must_use]
    fn new(failures: Vec<RetryFailureRecord>, final_class: FailureClass) -> Self {
        Self::with_trail(failures, final_class, None)
    }

    #[must_use]
    pub(crate) fn with_last_raw(
        failures: Vec<RetryFailureRecord>,
        final_class: FailureClass,
        last_raw: Option<String>,
    ) -> Self {
        Self::with_trail(failures, final_class, last_raw)
    }

    #[must_use]
    fn shutdown(failures: Vec<RetryFailureRecord>) -> Self {
        Self::new(failures, FailureClass::Shutdown)
    }
}

impl fmt::Display for RetryExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.detail)
    }
}

impl std::error::Error for RetryExhausted {}

/// Record the failure telemetry and return the terminal `Err` — the shared hard-fail tail of every
/// retry-loop exit. Fail-open loops (research synthesis partial output, consensus repair fallback)
/// deliberately bypass this helper.
pub(crate) fn fail_exhausted<T>(
    request: &ChatRequest,
    operation_started: Instant,
    exhausted: RetryExhausted,
) -> Result<T, RetryExhausted> {
    crate::stats::record_llm_failure(request, operation_started, &exhausted);
    Err(exhausted)
}

// Mutable per-operation state shared by the outer retry loops: the backoff schedule, the
// per-attempt failure trail and Retry-After stickiness, so all loops cannot drift.
pub(crate) struct RetryLoop {
    policy: RetryPolicy,
    backoffs: Vec<u64>,
    failures: Vec<RetryFailureRecord>,
    last_retry_after: Option<u64>,
    /// 1-based index the NEXT recorded failure gets. Only real failed attempts advance it —
    /// `record_trail_only` appends do not, so a trail-only entry cannot inflate a later row's
    /// index.
    attempts_failed: u32,
    /// `llm_failures` persistence context — `None` when the request carries no metadata (the gating
    /// the `llm_requests` rows use) or when the loop was built without a request (tests).
    operation: Option<crate::stats::LlmOperationCtx>,
    /// Start of the in-flight attempt — seeds the next failure row's per-attempt duration.
    attempt_started: Option<Instant>,
}

impl RetryLoop {
    /// Start a metadata-less operation (no `llm_failures` rows) — the pure schedule/trail core
    /// behind [`RetryLoop::new_scoped`], also used by the schedule/trail tests.
    #[must_use]
    fn new(policy: &RetryPolicy) -> Self {
        Self {
            policy: policy.clone(),
            backoffs: backoff_sequence(policy),
            failures: Vec::new(),
            last_retry_after: None,
            attempts_failed: 0,
            operation: None,
            attempt_started: None,
        }
    }

    /// Start a scoped operation: per-attempt failures persist to `llm_failures` when the request
    /// carries metadata (the same gating as the `llm_requests` row).
    #[must_use]
    pub(crate) fn new_scoped(policy: &RetryPolicy, request: &ChatRequest) -> Self {
        Self {
            operation: crate::stats::LlmOperationCtx::from_request(request),
            ..Self::new(policy)
        }
    }

    #[must_use]
    pub(crate) fn into_failures(self) -> Vec<RetryFailureRecord> {
        self.failures
    }

    #[must_use]
    pub(crate) fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// Mark the start of one attempt with the caller's instant — seeds the duration on the next
    /// failure row (absent → NULL `attempt_duration_ms`).
    pub(crate) fn begin_attempt(&mut self, started: Instant) {
        self.attempt_started = Some(started);
    }

    /// Record a failed attempt: update the sticky Retry-After and persist one `llm_failures` row
    /// (fail-open). A record without a Retry-After (parse / NoResponse) clears any stale value from
    /// an earlier attempt. `attempt` is the 1-based sequential failure index within the operation.
    pub(crate) fn record(&mut self, rec: RetryFailureRecord) {
        self.last_retry_after = rec.retry_after_ms;
        self.attempts_failed = self.attempts_failed.saturating_add(1);
        let attempt_started = self.attempt_started.take();
        if let Some(operation) = self.operation.as_ref() {
            crate::stats::record_llm_attempt_failure(
                operation,
                self.attempts_failed,
                &rec,
                attempt_started,
            );
        }
        self.failures.push(rec);
    }

    /// Append a trail-only failure record — no `llm_failures` row. For synthetic entries that do
    /// not correspond to a failed HTTP attempt (the consensus fallback keeps its trail non-empty
    /// this way).
    pub(crate) fn record_trail_only(&mut self, rec: RetryFailureRecord) {
        self.last_retry_after = rec.retry_after_ms;
        self.failures.push(rec);
    }

    /// Sleep between attempts, honouring the schedule and Retry-After;
    /// `Err(FailureClass::Shutdown)` when the global shutdown token fires during the sleep.
    /// `attempt` is the backoff-schedule index (`attempt - 1`) and the guard trigger (no sleep when
    /// `attempt >= max_attempts`). Callers indexing by a separate counter (e.g. consecutive
    /// transport failures) must add their own round-based guard.
    pub(crate) async fn sleep_between(&self, attempt: u32) -> Result<(), FailureClass> {
        if attempt >= self.policy.max_attempts {
            return Ok(());
        }
        let schedule_ms = self.backoffs[(attempt - 1) as usize];
        let sleep_ms = compute_sleep_ms(schedule_ms, self.last_retry_after);
        if !crate::shutdown::sleep_or_shutdown(Duration::from_millis(sleep_ms)).await {
            return Err(FailureClass::Shutdown);
        }
        Ok(())
    }

    /// Final classification when attempts are exhausted: the last recorded failure's class (or
    /// [`FailureClass::Transport`] if none was recorded).
    #[must_use]
    pub(crate) fn final_class(&self) -> FailureClass {
        self.failures
            .last()
            .map_or(FailureClass::Transport, |r| r.class)
    }
}

// The agent-loop LLM call with the outer retry loop. Byte-identical request across ALL attempts;
// retries provider failures of any retryable class (the 600 s idle timeout is the only per-attempt
// bound). Any `Ok` response is accepted — empty text is a valid tool-call turn. A reasoning-only
// stop is NOT handled here: the caller classifies and recovers it via bounded continuation before
// any persistence or display.
pub(crate) async fn agent_chat(
    request: ChatRequest,
    policy: &RetryPolicy,
) -> Result<ChatResponse, RetryExhausted> {
    let mut loop_state = RetryLoop::new_scoped(policy, &request);
    let operation_started = Instant::now();

    for attempt in 1..=policy.max_attempts {
        loop_state.begin_attempt(Instant::now());
        match crate::providers::chat_scoped(request.clone()).await {
            Ok(resp) => {
                crate::stats::record_llm_success(&request, operation_started, attempt, &resp);
                return Ok(resp);
            }
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(err.record);
                if non_retryable {
                    let exhausted = RetryExhausted::new(loop_state.into_failures(), err.class);
                    return fail_exhausted(&request, operation_started, exhausted);
                }
            }
        }

        if let Err(FailureClass::Shutdown) = loop_state.sleep_between(attempt).await {
            let exhausted = RetryExhausted::shutdown(loop_state.into_failures());
            return fail_exhausted(&request, operation_started, exhausted);
        }
    }

    // Exhausted — report the last failure's class.
    let final_class = loop_state.final_class();
    let exhausted = RetryExhausted::new(loop_state.into_failures(), final_class);
    fail_exhausted(&request, operation_started, exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_backoff_sequence_doubles_capped() {
        let p = RetryPolicy {
            max_attempts: 6,
            base_backoff_ms: 10_000,
            max_backoff_ms: 30_000,
        };
        assert_eq!(
            backoff_sequence(&p),
            vec![10_000, 20_000, 30_000, 30_000, 30_000]
        );
    }

    /// A `Retry-After` is followed within the policy's own window, and without one the sleep
    /// jitters inside ±25% of the base.
    #[test]
    fn compute_sleep_honors_retry_after_and_jitters_within_bounds() {
        assert_eq!(compute_sleep_ms(5_000, Some(7_000)), 7_000);
        assert_eq!(compute_sleep_ms(5_000, Some(1_000)), 5_000);
        assert_eq!(compute_sleep_ms(5_000, Some(200_000)), 60_000);

        for _ in 0..200 {
            let v = compute_sleep_ms(10_000, None);
            assert!(
                (7_500..12_500).contains(&v),
                "jitter out of ±25% for base 10000: {v}"
            );
        }
    }

    #[tokio::test]
    async fn stale_retry_after_does_not_stick_across_failures() {
        // A 429 Retry-After applies only to the sleep following the 429. A later failure without
        // one must clear it — otherwise a stale value wastes up to 60 s of the retry budget per
        // occurrence.
        let policy = tiny_test_policy();
        let mut loop_state = RetryLoop::new(&policy);

        // 429-style failure carries a Retry-After.
        let rec = RetryFailureRecord::new_simple(
            FailureClass::Transport,
            &anyhow::anyhow!("429 rate limited"),
            Some(60_000),
        );
        loop_state.record(rec);
        assert_eq!(loop_state.last_retry_after, Some(60_000));

        // A later parse/NoResponse failure has no Retry-After → clears it.
        let rec = RetryFailureRecord::new_simple(
            FailureClass::NoResponse,
            &anyhow::anyhow!("empty response"),
            None,
        );
        loop_state.record(rec);
        assert_eq!(
            loop_state.last_retry_after, None,
            "stale Retry-After must not stick to later sleeps"
        );
    }

    #[test]
    fn failure_class_labels_and_retryability() {
        assert!(FailureClass::Transport.is_retryable());
        assert!(FailureClass::TruncatedEnvelope.is_retryable());
        assert!(FailureClass::Parse.is_retryable());
        assert!(FailureClass::OutOfRangeScore.is_retryable());
        assert!(!FailureClass::NonRetryable.is_retryable());
        assert!(!FailureClass::Shutdown.is_retryable());
        assert_eq!(
            FailureClass::TruncatedEnvelope.label(),
            "truncated_envelope"
        );
    }

    #[test]
    #[serial_test::serial(provider)] // synthesis() applies the process-global test-policy override
    fn defaults_are_hardcoded() {
        // No config surface exists — the policy must always be the hardcoded defaults, whatever
        // stray `config_kv` rows say.
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, DEFAULT_RETRY_MAX_ATTEMPTS);
        assert_eq!(policy.base_backoff_ms, DEFAULT_RETRY_BASE_BACKOFF_MS);
        assert_eq!(policy.max_backoff_ms, DEFAULT_RETRY_MAX_BACKOFF_MS);
        let synthesis = RetryPolicy::synthesis();
        assert_eq!(synthesis.max_attempts, DEFAULT_SYNTHESIS_MAX_ATTEMPTS);
        assert_eq!(synthesis.base_backoff_ms, DEFAULT_SYNTHESIS_BASE_BACKOFF_MS);
        assert_eq!(synthesis.max_backoff_ms, DEFAULT_SYNTHESIS_MAX_BACKOFF_MS);
    }

    #[tokio::test]
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    async fn agent_chat_rides_out_sustained_outage_and_recovers() {
        // The binding agent-loop budget (13 attempts): the first 12 attempts hit a sustained
        // 503-style outage, the 13th recovers.
        let policy = RetryPolicy {
            max_attempts: 13,
            base_backoff_ms: 1,
            max_backoff_ms: 1,
        };
        let mut fake = crate::util::test::FakeProvider::new();
        for i in 0..12 {
            fake = fake.err(FailureClass::Transport, &format!("503 outage attempt {i}"));
        }
        let fake = fake.ok("recovered");
        let _provider_guard = crate::util::test::install_fake_provider(std::sync::Arc::new(fake));
        let request = crate::providers::test_request(vec![crate::ChatMessage::user("hi")], None);
        let resp = agent_chat(request, &policy)
            .await
            .expect("must recover on attempt 13");
        assert_eq!(resp.text_or_empty(), "recovered");
    }
}
