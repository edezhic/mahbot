//! Outer retry orchestration for LLM operations.
//!
//! The expensive LLM paths retry through this module — agent-loop chat calls,
//! structured extraction (verdicts, diagnostics discovery, comment summaries,
//! research orchestration), grouping repair (analyze consolidation, joint-verdict
//! synthesis) and deep-research synthesis — so every chat/extraction retry
//! budget lives in one place. The list is intentionally non-exhaustive:
//! behavior is defined by the policy constructors, not by per-path
//! documentation.
//!
//! # Single retry authority
//!
//! The outer loop in this module is the **single retry authority**: on these
//! calls provider-internal retries are suppressed (see [`Provider::chat_scoped`]),
//! so total provider HTTP calls per operation are explicitly bounded by the
//! policy in effect (see "Schedules").
//!
//! # Byte-identical retry parameters
//!
//! All request parameters (model, messages, tools, max_tokens, reasoning_effort,
//! provider routing) are byte-identical across ALL attempts — reasoning_effort
//! is FIXED (prefix cache preservation — never lower it), no escalation of any
//! kind. The only permitted change is appended feedback on a failed attempt —
//! the parse-failure re-prompt in the extraction path, repair-round
//! instructions, synthesis feedback — which extends the cached prefix.
//!
//! # Schedules
//!
//! Four policies, snapshotted once at operation start ([`RetryPolicy`]). All
//! share the same mechanics: ±25% jitter on the SLEEP ONLY (never on request
//! bytes); Retry-After honored, clamped [5000 ms, 60000 ms];
//! shutdown-abortable. They differ ONLY in attempt counts and backoff — no
//! per-operation wall-clock cap exists; the only timing bound is the 600 s
//! idle timeout ([`DEFAULT_IDLE_TIMEOUT`]). A stalled attempt is cut at
//! 600 s, so worst case is roughly `attempts × 600 s + backoff` (a body that
//! keeps trickling bytes is never cut).
//!
//! - **Default** ([`RetryPolicy::current`]) — 13 attempts, backoff
//!   5/10/20/40/60/60… s (base 5000 ms, doubling capped at 60000 ms; total
//!   sleep 555 s) — rides out sustained 503 outages before failing.
//! - **Synthesis** ([`RetryPolicy::synthesis`]) — 3 total attempts (1
//!   synthesis + up to 2 repair rounds), 30–45 s backoff band; bounded so a
//!   bad grouping degrades to the deterministic fallback after at most 3
//!   calls.
//! - **Comment** ([`RetryPolicy::comment`]) — 3 attempts, default backoff;
//!   for fail-open comment-only extraction, where a long retry would stall
//!   the pipeline for a non-critical operation.
//! - **Continuation** ([`RetryPolicy::continuation`]) — 3 attempts, no
//!   inter-attempt sleep; the reasoning-only-stop recovery budget
//!   (appended-only re-requests, see
//!   [`crate::agent::Agent::recover_reasoning_only_stop`]). Retryable
//!   transport errors re-send the byte-identical request; non-retryable
//!   errors break immediately.
//!
//! Per-attempt timeout semantics come from [`Provider::chat_scoped`]: the
//! header wait (TTFB) and each body-read chunk wait are bounded by the 600 s
//! idle timeout, resetting while data flows — a healthy-but-slow generation
//! with data flowing is never cut, but a stalled attempt (no bytes for 600 s)
//! is aborted and retried per the policy in effect.
//!
//! # Telemetry
//!
//! Every failed attempt produces a [`RetryFailureRecord`] appended to the
//! in-memory failure trail; terminal exhaustion surfaces it through
//! [`RetryExhausted`], which feeds the live `llm_requests` stats rows
//! (retry_attempts / finish_reason / failure_class). Nothing is persisted
//! per-attempt — the trail lives only in memory for the operation's lifetime.

use std::fmt;
use std::time::{Duration, Instant};

#[cfg(test)]
use crate::util::UnwrapPoison;
use crate::{ChatRequest, ChatResponse};

// ── Hardcoded retry defaults (no config surface — fixed in code) ─────────

pub(crate) const DEFAULT_RETRY_MAX_ATTEMPTS: u32 = 13;
pub(crate) const DEFAULT_RETRY_BASE_BACKOFF_MS: u64 = 5_000;
pub(crate) const DEFAULT_RETRY_MAX_BACKOFF_MS: u64 = 60_000;

/// Dedicated joint-verdict synthesis retry schedule: total calls (1 full
/// synthesis + up to N-1 repair rounds; default 3 = the lower edge of the
/// approved 3–5 band), 30–45 s backoff band (base 30 s, cap 45 s, ±25%
/// jitter on sleeps).
pub(crate) const DEFAULT_SYNTHESIS_MAX_ATTEMPTS: u32 = 3;
pub(crate) const DEFAULT_SYNTHESIS_BASE_BACKOFF_MS: u64 = 30_000;
pub(crate) const DEFAULT_SYNTHESIS_MAX_BACKOFF_MS: u64 = 45_000;

/// Dedicated reasoning-only-stop continuation schedule: up to 3 appended-only
/// continuation re-requests after the original in-class response, bounded by
/// the attempt count (the [`RetryPolicy::comment`] precedent) so a stuck
/// reasoning-only model fails the turn safely. Each continuation attempt is a
/// single `chat_scoped` call (no inner transport retry — the appended tail
/// makes every new reasoning state a fresh request; retryable transport errors
/// re-send the identical bytes).
pub(crate) const DEFAULT_CONTINUATION_MAX_ATTEMPTS: u32 = 3;

/// The single timeout governing every LLM chat request — bounds the
/// response-header wait (TTFB) and each body-read chunk wait, resetting while
/// data flows. A stalled attempt is aborted at 600 s and retried per the retry
/// policy in effect; a healthy-but-slow generation with data flowing is never
/// cut. No total request timeout applies to chat attempts.
pub(crate) const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Lower clamp for Retry-After honoring.
const RETRY_AFTER_MIN_MS: u64 = 5_000;
/// Upper clamp for Retry-After honoring — unified at 60 s across all scoped
/// retry paths.
pub(crate) const RETRY_AFTER_MAX_MS: u64 = 60_000;

// ── RetryPolicy — snapshot of tunables at operation start ────────────────

/// Snapshot of retry tunables taken at operation start.
///
/// Hot-reload mid-retry must not change the schedule — callers snapshot once
/// and reuse. Invalid configured values fall back to defaults.
#[derive(Debug, Clone)]
pub(crate) struct RetryPolicy {
    /// Total loop attempts (default 13). The operation makes at most this many
    /// provider HTTP calls.
    pub max_attempts: u32,
    /// Base backoff in milliseconds (default 5000).
    pub base_backoff_ms: u64,
    /// Backoff cap in milliseconds (default 60000).
    pub max_backoff_ms: u64,
}

impl RetryPolicy {
    /// Build the default policy from the hardcoded constants.
    #[must_use]
    pub(crate) fn default() -> Self {
        Self {
            max_attempts: DEFAULT_RETRY_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_RETRY_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_RETRY_MAX_BACKOFF_MS,
        }
    }

    /// Resolve the policy for a scoped operation.
    #[must_use]
    pub(crate) fn current() -> Self {
        #[cfg(test)]
        if let Some(p) = test_override() {
            return p;
        }
        Self::default()
    }

    /// Build the joint-verdict synthesis policy from the hardcoded constants.
    /// `synthesis_max_attempts` is the TOTAL call count: 1 full
    /// synthesis + up to N-1 repair rounds (default 3 — the lower edge of the
    /// approved 3–5 band). The synthesis loop is deliberately bounded so a
    /// bad grouping pass degrades to the deterministic fallback comment
    /// after at most 3 calls.
    #[must_use]
    pub(crate) fn synthesis() -> Self {
        #[cfg(test)]
        if let Some(p) = test_override() {
            return p;
        }
        Self {
            max_attempts: DEFAULT_SYNTHESIS_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_SYNTHESIS_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_SYNTHESIS_MAX_BACKOFF_MS,
        }
    }

    /// Build the comment-only extraction policy for fail-open callers: the
    /// caller falls back to raw text on failure, so a bounded attempt count
    /// avoids stalling the pipeline for a non-critical operation (the
    /// 13-attempt budget is for verdict gates).
    #[must_use]
    pub(crate) fn comment() -> Self {
        #[cfg(test)]
        if let Some(p) = test_override() {
            return p;
        }
        Self {
            max_attempts: 3,
            base_backoff_ms: DEFAULT_RETRY_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_RETRY_MAX_BACKOFF_MS,
        }
    }

    /// Build the reasoning-only-stop continuation policy: bounded recovery for
    /// an empty-content/no-tool response (see
    /// [`crate::agent::Agent::recover_reasoning_only_stop`]). Mirrors the
    /// [`Self::comment`] budget — 3 attempts. No inter-attempt sleep: the
    /// appended-only tail makes each new reasoning
    /// state a fresh request with a byte-stable prefix (retryable transport
    /// errors re-send the identical bytes; non-retryable errors break
    /// immediately).
    /// `base_backoff_ms`/`max_backoff_ms` are unused (the manual continuation
    /// loop never sleeps between attempts).
    #[must_use]
    pub(crate) fn continuation() -> Self {
        #[cfg(test)]
        if let Some(p) = test_override() {
            return p;
        }
        Self {
            max_attempts: DEFAULT_CONTINUATION_MAX_ATTEMPTS,
            base_backoff_ms: 0,
            max_backoff_ms: 0,
        }
    }
}

/// Test seam: install a tiny retry policy so scoped-retry tests run fast.
/// Guarded by `util::test::retry_tests_lock()` in tests that use it.
#[cfg(test)]
static TEST_POLICY_OVERRIDE: std::sync::RwLock<Option<RetryPolicy>> = std::sync::RwLock::new(None);

/// In tests, the override installed via [`swap_test_retry_policy`] takes
/// precedence so retry-loop tests don't sleep for minutes; otherwise the
/// hardcoded defaults apply. Poison-tolerant like the other test seams
/// ([`crate::util::test::retry_tests_lock`]): a failing test must not
/// cascade into later ones.
#[cfg(test)]
fn test_override() -> Option<RetryPolicy> {
    let guard = TEST_POLICY_OVERRIDE.read().unwrap_poison();
    guard.as_ref().cloned()
}

/// Swap the test retry-policy override, returning the previous value so an
/// RAII guard (see `util::test::RetryPolicyGuard`) can restore it on drop —
/// including during a panic. Mirrors
/// [`crate::providers::swap_provider_for_test`].
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

/// Compute the canonical backoff sleep sequence (ms) for a policy.
///
/// Pure doubling from `base_backoff_ms`, capped at `max_backoff_ms`. The
/// hardcoded defaults (base 5000 / cap 60000) yield the binding agent-loop
/// schedule 5/10/20/40/60/60… s — total sleep 555 s over 13 attempts.
#[must_use]
pub(crate) fn backoff_sequence(policy: &RetryPolicy) -> Vec<u64> {
    let sleeps = policy.max_attempts.saturating_sub(1) as usize;
    let mut seq = Vec::with_capacity(sleeps);
    for i in 0..sleeps {
        let doubled = policy.base_backoff_ms.saturating_mul(1u64 << i.min(6));
        seq.push(doubled.min(policy.max_backoff_ms));
    }
    seq
}

/// ±25% jitter around `base_ms`: random within [75%, 125%) of base so
/// parallel agents retrying on the same transient error don't synchronize.
/// Jitter touches the SLEEP ONLY, never the request bytes.
/// The modulo guard keeps 0/1 ms schedules (test policies) division-safe.
#[must_use]
pub(crate) fn jittered_backoff_ms(base_ms: u64) -> u64 {
    base_ms - base_ms / 4 + (rand::random::<u64>() % (base_ms / 2).max(1))
}

/// Compute the actual sleep for one inter-attempt gap.
///
/// Retry-After (when present) is authoritative and followed precisely, clamped
/// to [5000 ms, 60000 ms]. Otherwise ±25% jitter is applied to the schedule
/// backoff — jitter touches the SLEEP ONLY, never the request bytes.
#[must_use]
pub(crate) fn compute_sleep_ms(schedule_ms: u64, retry_after_ms: Option<u64>) -> u64 {
    if let Some(ra) = retry_after_ms {
        ra.clamp(RETRY_AFTER_MIN_MS, RETRY_AFTER_MAX_MS)
    } else {
        jittered_backoff_ms(schedule_ms)
    }
}

// ── Failure classification ───────────────────────────────────────────────

/// Granular failure classification for the retry/error trail.
///
/// The `Membership`/`Completeness`/`ContradictionAgents`/`ValidationOther`
/// variants carry repair-validation semantics on the shared provider-failure
/// path (granular causes ride the existing failure-cause field); they are
/// retryable and only ever produced by the grouping core. `TruncatedOutput`
/// marks a provider output-token-limit truncation (finish_reason "length") —
/// retryable with shorten/compress feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailureClass {
    /// Network/transport error (connection reset, timeouts, 5xx, 429).
    Transport,
    /// Server truncated the response envelope: body-read error or
    /// content-length mismatch (the "error reading response body" /
    /// "EOF while parsing" defect class).
    TruncatedEnvelope,
    /// Response body read fully but JSON did not parse (LLM format issue).
    Parse,
    /// Parsed successfully but a validation hook rejected the value
    /// (e.g. verdict score outside [0,10]).
    OutOfRangeScore,
    /// Repair-round group rejected: a member's item id is out of range, or
    /// the member was already placed in a frozen group / pinned by an
    /// accepted contradiction reference.
    Membership,
    /// Repair-round proposal silently dropped unfrozen items (incomplete
    /// coverage).
    Completeness,
    /// A proposed contradiction/reference lacks ≥2 distinct cited agents.
    ContradictionAgents,
    /// A validation rejection outside the granular categories (empty
    /// heading/summary, malformed structure, out-of-range group reference).
    ValidationOther,
    /// Provider returned text cut at its output-token limit (finish_reason
    /// "length").
    TruncatedOutput,
    /// Provider returned an empty text response.
    NoResponse,
    /// Permanent client error (auth, quota, invalid model, tool schema).
    NonRetryable,
    /// Global shutdown fired mid-operation.
    Shutdown,
}

impl FailureClass {
    /// Whether the operation should keep retrying after this failure.
    #[must_use]
    pub(crate) const fn is_retryable(self) -> bool {
        !matches!(self, Self::NonRetryable | Self::Shutdown)
    }

    /// Short stable label for logs / telemetry.
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

// ── Per-attempt failure diagnostics ──────────────────────────────────────

/// Per-attempt failure diagnostics, carried in the in-memory retry trail.
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

// ── RetryExhausted — terminal error carrying the full trail ──────────────

/// Terminal failure of an outer retry loop.
///
/// Carries the per-attempt diagnostics trail plus the final classification so
/// callers can (a) fail open with a useful reason (consolidation) or (b) write
/// the last-attempt raw text into a ticket comment (verdict extraction).
///
/// `last_raw` is `Some(text)` when the last COMPLETED attempt produced
/// assistant text (including `Some("")` for a tool-call final attempt) and
/// `None` when that attempt died before producing text (transport /
/// truncation / budget failure) — text from earlier attempts is never
/// presented as the last attempt.
#[derive(Debug, Clone)]
pub(crate) struct RetryExhausted {
    pub failures: Vec<RetryFailureRecord>,
    pub final_class: FailureClass,
    /// Last-attempt raw text (verdict-extraction ticket-comment dumps). See the
    /// type docs for the precise "last completed attempt" semantics.
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

    /// Terminal error carrying the last-attempt raw text (verdict extraction).
    #[must_use]
    pub(crate) fn with_last_raw(
        failures: Vec<RetryFailureRecord>,
        final_class: FailureClass,
        last_raw: Option<String>,
    ) -> Self {
        Self::with_trail(failures, final_class, last_raw)
    }

    #[must_use]
    pub(crate) fn shutdown(failures: Vec<RetryFailureRecord>) -> Self {
        Self::new(failures, FailureClass::Shutdown)
    }
}

impl fmt::Display for RetryExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.detail)
    }
}

impl std::error::Error for RetryExhausted {}

/// Record the failure telemetry and return the terminal `Err` — the shared
/// hard-fail tail of every retry-loop exit. Fail-open loops (research
/// synthesis partial output, consensus repair fallback) record then continue,
/// so they deliberately bypass this helper.
pub(crate) async fn fail_exhausted<T>(
    request: &ChatRequest,
    operation_started: Instant,
    exhausted: RetryExhausted,
) -> Result<T, RetryExhausted> {
    crate::stats::record_llm_failure(request, operation_started, &exhausted).await;
    Err(exhausted)
}

// ── Shared retry-loop state (schedule + failure trail) ───────────────────

/// Mutable per-operation state shared by the outer retry loops.
///
/// Encapsulates the backoff schedule, the per-attempt failure trail, and
/// Retry-After stickiness so all outer retry loops cannot drift — schedule and
/// trail mechanics live in exactly one place.
pub(crate) struct RetryLoop {
    policy: RetryPolicy,
    backoffs: Vec<u64>,
    failures: Vec<RetryFailureRecord>,
    last_retry_after: Option<u64>,
}

impl RetryLoop {
    /// Start a new operation.
    #[must_use]
    pub(crate) fn new(policy: &RetryPolicy) -> Self {
        Self {
            policy: policy.clone(),
            backoffs: backoff_sequence(policy),
            failures: Vec::new(),
            last_retry_after: None,
        }
    }

    /// Take ownership of the failure trail (terminal paths).
    #[must_use]
    pub(crate) fn into_failures(self) -> Vec<RetryFailureRecord> {
        self.failures
    }

    /// True when at least one failure was recorded this operation.
    #[must_use]
    pub(crate) fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// Record a failed attempt, updating the sticky Retry-After. A record
    /// without a Retry-After (parse / NoResponse failures) clears any stale
    /// value from an earlier attempt, so a 429 Retry-After never bleeds into
    /// later sleeps.
    pub(crate) fn record(&mut self, rec: RetryFailureRecord) {
        self.last_retry_after = rec.retry_after_ms;
        self.failures.push(rec);
    }

    /// Sleep between attempts, honoring the schedule / Retry-After. Returns
    /// `Err(FailureClass::Shutdown)` when the global shutdown token fires
    /// during the sleep.
    ///
    /// `attempt` is the backoff-schedule index (`attempt - 1`) and the guard
    /// trigger (no sleep when `attempt >= max_attempts`), normally the 1-based
    /// index of the attempt that just completed. Callers indexing by a
    /// separate counter (e.g. consecutive transport failures) must add their
    /// own round-based guard — the final-attempt skip does not fire for them.
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

    /// Final classification when attempts are exhausted: the last recorded
    /// failure's class (or [`FailureClass::Transport`] if none was recorded).
    #[must_use]
    pub(crate) fn final_class(&self) -> FailureClass {
        self.failures
            .last()
            .map_or(FailureClass::Transport, |r| r.class)
    }
}

// ── The agent retry loop ────────────────────────────────────────────────

/// Agent-loop LLM call with the outer retry loop.
///
/// Byte-identical request across ALL attempts; retries provider failures of
/// any retryable class (the 600 s idle timeout is the only per-attempt bound).
/// Any `Ok` response is accepted — empty text is a valid tool-call turn. A
/// reasoning-only stop (empty text, no tool calls) is NOT handled here: the
/// agent-loop caller classifies and recovers it via bounded continuation
/// ([`crate::agent::Agent::recover_reasoning_only_stop`]) before any
/// persistence/display.
pub(crate) async fn agent_chat(
    request: ChatRequest,
    policy: &RetryPolicy,
) -> Result<ChatResponse, RetryExhausted> {
    let mut loop_state = RetryLoop::new(policy);
    let operation_started = Instant::now();

    for attempt in 1..=policy.max_attempts {
        match crate::providers::chat_scoped(request.clone()).await {
            Ok(resp) => {
                crate::stats::record_llm_success(&request, operation_started, attempt, &resp).await;
                return Ok(resp);
            }
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(err.record);
                if non_retryable {
                    let exhausted = RetryExhausted::new(loop_state.into_failures(), err.class);
                    return fail_exhausted(&request, operation_started, exhausted).await;
                }
            }
        }

        if let Err(FailureClass::Shutdown) = loop_state.sleep_between(attempt).await {
            let exhausted = RetryExhausted::shutdown(loop_state.into_failures());
            return fail_exhausted(&request, operation_started, exhausted).await;
        }
    }

    // Exhausted — report the last failure's class.
    let final_class = loop_state.final_class();
    let exhausted = RetryExhausted::new(loop_state.into_failures(), final_class);
    fail_exhausted(&request, operation_started, exhausted).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backoff_sequence_is_doubling_capped() {
        let p = RetryPolicy {
            max_attempts: DEFAULT_RETRY_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_RETRY_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_RETRY_MAX_BACKOFF_MS,
        };
        assert_eq!(
            backoff_sequence(&p),
            vec![
                5_000, 10_000, 20_000, 40_000, 60_000, 60_000, 60_000, 60_000, 60_000, 60_000,
                60_000, 60_000
            ]
        );
    }

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

    #[test]
    fn compute_sleep_honors_retry_after_clamped() {
        // Retry-After followed precisely within [5000, 60000].
        assert_eq!(compute_sleep_ms(5_000, Some(7_000)), 7_000);
        assert_eq!(compute_sleep_ms(5_000, Some(1_000)), 5_000);
        assert_eq!(compute_sleep_ms(5_000, Some(200_000)), 60_000);
    }

    #[test]
    fn compute_sleep_jitter_within_25_percent() {
        for _ in 0..200 {
            let v = compute_sleep_ms(10_000, None);
            assert!(
                (7_500..12_500).contains(&v),
                "jitter out of ±25% for base 10000: {v}"
            );
        }
    }

    #[test]
    fn stale_retry_after_does_not_stick_across_failures() {
        // A 429 Retry-After applies only to the sleep following the 429. A
        // later non-429 failure (parse / NoResponse, no Retry-After) must
        // clear it — otherwise a stale value wastes up to 60 s of the retry
        // budget per occurrence (reviewer finding).
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
    fn defaults_are_hardcoded() {
        let _guard = crate::util::test::retry_tests_lock();
        // No config surface exists — the policy must always be the
        // hardcoded defaults regardless of any stray config_kv rows.
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
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn agent_chat_rides_out_sustained_outage_and_recovers() {
        let _guard = crate::util::test::retry_tests_lock();
        // The binding agent-loop budget (13 attempts): the first 12 attempts
        // hit a sustained 503-style outage, the 13th recovers.
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
