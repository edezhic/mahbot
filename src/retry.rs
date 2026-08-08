//! Outer retry orchestration for LLM operations.
//!
//! Four expensive code paths use this module:
//!
//! 1. **Agent-loop LLM calls** — every chat call an agent makes while working
//!    (`src/agent.rs`).
//! 2. **Verdict extraction** — structured pass/fail verdicts from Analyst /
//!    Reviewer / QA / Sanitation agents (`src/extraction.rs`).
//! 3. **Diagnostics discovery** — workspace diagnostics command extraction
//!    (`src/workspace.rs`).
//! 4. **Analyst consolidation** — the synthesis of the 3 parallel analyst
//!    reports in the ask tool (`src/tools/ask.rs`).
//!
//! # Single retry authority
//!
//! The outer loop in this module is the **single retry authority**: on these
//! calls provider-internal retries are suppressed (see [`Provider::chat_scoped`]),
//! so total provider HTTP calls per operation are explicitly bounded:
//! 13 per agent LLM call, 13 per verdict extraction, 13 per diagnostics
//! discovery, 13 per consolidation.
//!
//! # Byte-identical retry parameters
//!
//! All request parameters (model, messages, tools, max_tokens, reasoning_effort,
//! provider routing) are byte-identical across ALL attempts — reasoning_effort
//! is FIXED (prefix cache preservation — never lower it), no escalation of any
//! kind. The only
//! permitted change is the parse-failure re-prompt in the extraction path
//! (appended messages), which extends the cached prefix.
//!
//! # Schedule
//!
//! 13 loop attempts; backoff 5/10/20/40/60/60… s (base 5000 ms, doubling
//! capped at 60000 ms; total sleep 555 s); ±25% jitter on the SLEEP ONLY
//! (never on request bytes); Retry-After honored, clamped [5000 ms, 60000 ms];
//! shutdown-abortable; per-operation wall-clock cap 720 s (12 min),
//! authoritative over attempt count — rides out sustained 503 outages up to
//! ~10 min before failing (bounded worst-case stall 12 min).
//!
//! Per-attempt timeout semantics come from [`Provider::chat_scoped`]: the
//! header wait (TTFB) is bounded by the 1-min idle timeout and the whole
//! attempt by the remaining operation budget — a healthy-but-slow generation
//! with data flowing is never cut, but a pre-header stall longer than 60 s is
//! aborted and retried.
//!
//! # Telemetry
//!
//! Every failed attempt produces a [`RetryFailureRecord`] persisted to the
//! dedicated `retry_failures` table in the logs store (consolidated with the
//! tool-call stats tables). Persistence is best-effort: the logs store's
//! quarantine heals corruption only at boot, so mid-run logs corruption fails
//! consolidated stats writes too.

use std::fmt;
use std::time::{Duration, Instant};

use crate::config::CONFIG;
use crate::{ChatRequest, ChatResponse};

// ── Config defaults (shared with the `string_config_fields!` macro) ──────
// The string constants are referenced by config.rs's macro invocation
// (`or(...)` annotations); the parsed defaults live here.

pub(crate) const DEFAULT_RETRY_MAX_ATTEMPTS_STR: &str = "13";
pub(crate) const DEFAULT_RETRY_BASE_BACKOFF_MS_STR: &str = "5000";
pub(crate) const DEFAULT_RETRY_MAX_BACKOFF_MS_STR: &str = "60000";
pub(crate) const DEFAULT_OPERATION_TIMEOUT_SECS_STR: &str = "720";

pub(crate) const DEFAULT_RETRY_MAX_ATTEMPTS: u32 = 13;
pub(crate) const DEFAULT_RETRY_BASE_BACKOFF_MS: u64 = 5_000;
pub(crate) const DEFAULT_RETRY_MAX_BACKOFF_MS: u64 = 60_000;
pub(crate) const DEFAULT_OPERATION_TIMEOUT: Duration = Duration::from_mins(12);

/// Dedicated joint-verdict synthesis retry schedule: total calls (1 full
/// synthesis + up to N-1 repair rounds; default 3 = the lower edge of the
/// approved 3–5 band), 30–45 s backoff band (base 30 s, cap 45 s, ±25%
/// jitter on sleeps).
pub(crate) const DEFAULT_SYNTHESIS_MAX_ATTEMPTS_STR: &str = "3";
pub(crate) const DEFAULT_SYNTHESIS_BASE_BACKOFF_MS_STR: &str = "30000";
pub(crate) const DEFAULT_SYNTHESIS_MAX_BACKOFF_MS_STR: &str = "45000";
pub(crate) const DEFAULT_SYNTHESIS_MAX_ATTEMPTS: u32 = 3;
pub(crate) const DEFAULT_SYNTHESIS_BASE_BACKOFF_MS: u64 = 30_000;
pub(crate) const DEFAULT_SYNTHESIS_MAX_BACKOFF_MS: u64 = 45_000;

/// Idle (read) timeout for scoped calls — resets while data flows.
pub(crate) const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_mins(1);

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
    /// Whole-operation wall-clock cap (default 720 s). Authoritative over
    /// attempt count.
    pub operation_timeout: Duration,
    /// Idle timeout for a single provider call, resetting while data flows.
    pub idle_timeout: Duration,
}

impl RetryPolicy {
    /// Build from the global [`CONFIG`], falling back to defaults for invalid
    /// or missing values.
    #[must_use]
    pub(crate) fn from_config() -> Self {
        Self {
            max_attempts: parse_cfg_u64(
                "retry_max_attempts",
                &CONFIG.retry_max_attempts(),
                u64::from(DEFAULT_RETRY_MAX_ATTEMPTS),
            )
            .clamp(1, 100) as u32,
            base_backoff_ms: parse_cfg_u64(
                "retry_base_backoff_ms",
                &CONFIG.retry_base_backoff_ms(),
                DEFAULT_RETRY_BASE_BACKOFF_MS,
            )
            .clamp(1, 3_600_000),
            max_backoff_ms: parse_cfg_u64(
                "retry_max_backoff_ms",
                &CONFIG.retry_max_backoff_ms(),
                DEFAULT_RETRY_MAX_BACKOFF_MS,
            )
            .clamp(1, 3_600_000),
            operation_timeout: Duration::from_secs(
                parse_cfg_u64(
                    "operation_timeout_secs",
                    &CONFIG.operation_timeout_secs(),
                    DEFAULT_OPERATION_TIMEOUT.as_secs(),
                )
                .clamp(1, 86_400),
            ),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }

    /// Resolve the policy for a scoped operation.
    ///
    /// In tests, an override installed via [`swap_test_retry_policy`] takes
    /// precedence so retry-loop tests don't sleep for minutes; otherwise this
    /// reads the global [`CONFIG`].
    #[must_use]
    pub(crate) fn current() -> Self {
        #[cfg(test)]
        if let Ok(guard) = TEST_POLICY_OVERRIDE.read()
            && let Some(p) = guard.as_ref()
        {
            return p.clone();
        }
        Self::from_config()
    }

    /// Build the joint-verdict synthesis policy from the dedicated config
    /// keys (`synthesis_max_attempts`, `synthesis_base_backoff_ms`,
    /// `synthesis_max_backoff_ms`), falling back to defaults on invalid
    /// values. `synthesis_max_attempts` is the TOTAL call count: 1 full
    /// synthesis + up to N-1 repair rounds (default 3 — the lower edge of the
    /// approved 3–5 band). The synthesis loop is deliberately bounded so a
    /// bad grouping pass degrades to the deterministic fallback comment
    /// instead of burning minutes of wall time.
    ///
    /// Like [`Self::current`], a test override installed via
    /// [`swap_test_retry_policy`] takes precedence so synthesis tests run
    /// fast.
    #[must_use]
    pub(crate) fn synthesis_from_config() -> Self {
        #[cfg(test)]
        if let Ok(guard) = TEST_POLICY_OVERRIDE.read()
            && let Some(p) = guard.as_ref()
        {
            return p.clone();
        }
        Self {
            max_attempts: parse_cfg_u64(
                "synthesis_max_attempts",
                &CONFIG.synthesis_max_attempts(),
                u64::from(DEFAULT_SYNTHESIS_MAX_ATTEMPTS),
            )
            .clamp(1, 20) as u32,
            base_backoff_ms: parse_cfg_u64(
                "synthesis_base_backoff_ms",
                &CONFIG.synthesis_base_backoff_ms(),
                DEFAULT_SYNTHESIS_BASE_BACKOFF_MS,
            )
            .clamp(1, 3_600_000),
            max_backoff_ms: parse_cfg_u64(
                "synthesis_max_backoff_ms",
                &CONFIG.synthesis_max_backoff_ms(),
                DEFAULT_SYNTHESIS_MAX_BACKOFF_MS,
            )
            .clamp(1, 3_600_000),
            operation_timeout: Duration::from_mins(10),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}

/// Test seam: install a tiny retry policy so scoped-retry tests run fast.
/// Guarded by `util::test::retry_tests_lock()` in tests that use it.
#[cfg(test)]
static TEST_POLICY_OVERRIDE: std::sync::RwLock<Option<RetryPolicy>> = std::sync::RwLock::new(None);

/// Swap the test retry-policy override, returning the previous value so an
/// RAII guard (see `util::test::RetryPolicyGuard`) can restore it on drop —
/// including during a panic. Mirrors
/// [`crate::providers::swap_provider_for_test`].
#[cfg(test)]
pub(crate) fn swap_test_retry_policy(policy: RetryPolicy) -> Option<RetryPolicy> {
    let mut guard = TEST_POLICY_OVERRIDE
        .write()
        .expect("retry policy override poisoned");
    let previous = guard.take();
    *guard = Some(policy);
    previous
}

/// Restore a previously swapped-out test retry-policy override.
#[cfg(test)]
pub(crate) fn restore_test_retry_policy(previous: Option<RetryPolicy>) {
    *TEST_POLICY_OVERRIDE
        .write()
        .expect("retry policy override poisoned") = previous;
}

/// A tiny retry policy for tests: 3 attempts, ~1 ms backoff, 60 s wall cap.
#[cfg(test)]
pub(crate) fn tiny_test_policy() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 3,
        base_backoff_ms: 1,
        max_backoff_ms: 1,
        operation_timeout: Duration::from_mins(1),
        idle_timeout: Duration::from_secs(1),
    }
}

/// Parse a numeric config value, logging and falling back on invalid input.
fn parse_cfg_u64(key: &str, raw: &str, fallback: u64) -> u64 {
    match raw.trim().parse::<u64>() {
        Ok(v) if v > 0 => v,
        _ => {
            tracing::warn!(
                key,
                value = %raw,
                "Invalid retry config value — falling back to default {fallback}"
            );
            fallback
        }
    }
}

/// Compute the canonical backoff sleep sequence (ms) for a policy.
///
/// Pure doubling from `base_backoff_ms`, capped at `max_backoff_ms`. The
/// default configuration (base 5000 / cap 60000) yields the binding agent-loop
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
/// retryable and only ever produced by the grouping core.
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
    /// Repair-round group rejected: a member's text is not verbatim in the
    /// cited agent's list, the agent index is out of range, or the member
    /// was already placed in a frozen group.
    Membership,
    /// Repair-round proposal silently dropped unfrozen items (incomplete
    /// coverage).
    Completeness,
    /// A proposed contradiction/reference lacks ≥2 distinct cited agents.
    ContradictionAgents,
    /// A validation rejection outside the granular categories (empty
    /// heading/summary, malformed structure, out-of-range group reference).
    ValidationOther,
    /// Provider returned an empty text response.
    NoResponse,
    /// Permanent client error (auth, quota, invalid model, tool schema).
    NonRetryable,
    /// Global shutdown fired mid-operation.
    Shutdown,
    /// Operation wall-clock budget exhausted.
    WallClockExceeded,
}

impl FailureClass {
    /// Whether the operation should keep retrying after this failure.
    #[must_use]
    pub(crate) const fn is_retryable(self) -> bool {
        !matches!(
            self,
            Self::NonRetryable | Self::Shutdown | Self::WallClockExceeded
        )
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
            Self::NoResponse => "no_response",
            Self::NonRetryable => "non_retryable",
            Self::Shutdown => "shutdown",
            Self::WallClockExceeded => "wall_clock_exceeded",
        }
    }
}

// ── Per-attempt failure diagnostics ──────────────────────────────────────

/// Bytes of the response body head/tail captured on failure (each side).
pub(crate) const BODY_SNIPPET_BYTES: usize = 200;

/// Per-attempt failure diagnostics, persisted to the logs store's `retry_failures` table.
///
/// Captured as close to the HTTP boundary as possible so response metadata
/// survives stringification into the `anyhow` error trail.
#[derive(Debug, Clone)]
pub(crate) struct RetryFailureRecord {
    /// 1-based attempt number (stamped by the retry loop; providers emit 0).
    pub attempt: u32,
    /// Granular failure classification.
    pub class: FailureClass,
    /// Full error cause chain.
    pub error_chain: String,
    pub http_version: Option<String>,
    /// `Content-Length` response header, if present.
    pub content_length: Option<u64>,
    /// Actual body bytes read.
    pub actual_body_len: Option<usize>,
    pub content_encoding: Option<String>,
    pub transfer_encoding: Option<String>,
    /// Wall time consumed by this attempt.
    pub elapsed_ms: u64,
    /// First [`BODY_SNIPPET_BYTES`] bytes of the response body.
    pub body_head: String,
    /// Last [`BODY_SNIPPET_BYTES`] bytes of the response body.
    pub body_tail: String,
    /// `choices[0].finish_reason` parsed from the envelope (best effort).
    pub finish_reason: Option<String>,
    /// `usage.completion_tokens` parsed from the envelope (best effort).
    pub completion_tokens: Option<u64>,
    /// Retry-After value from the response, if any.
    pub retry_after_ms: Option<u64>,
    /// RFC3339 timestamp (set at construction).
    pub recorded_at: String,
}

impl RetryFailureRecord {
    /// Build a record from a plain error (no HTTP metadata available).
    #[must_use]
    pub(crate) fn new_simple(
        attempt: u32,
        class: FailureClass,
        error: &anyhow::Error,
        elapsed_ms: u64,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self {
            attempt,
            class,
            error_chain: format!("{error:#}"),
            http_version: None,
            content_length: None,
            actual_body_len: None,
            content_encoding: None,
            transfer_encoding: None,
            elapsed_ms,
            body_head: String::new(),
            body_tail: String::new(),
            finish_reason: None,
            completion_tokens: None,
            retry_after_ms,
            recorded_at: crate::turso::now(),
        }
    }

    /// Build a record with full response metadata.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) fn with_metadata(
        attempt: u32,
        class: FailureClass,
        error: &anyhow::Error,
        elapsed_ms: u64,
        http_version: Option<String>,
        content_length: Option<u64>,
        actual_body_len: Option<usize>,
        content_encoding: Option<String>,
        transfer_encoding: Option<String>,
        body: &str,
        finish_reason: Option<String>,
        completion_tokens: Option<u64>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        let (body_head, body_tail) = body_head_tail(body);
        Self {
            attempt,
            class,
            error_chain: format!("{error:#}"),
            http_version,
            content_length,
            actual_body_len,
            content_encoding,
            transfer_encoding,
            elapsed_ms,
            body_head,
            body_tail,
            finish_reason,
            completion_tokens,
            retry_after_ms,
            recorded_at: crate::turso::now(),
        }
    }
}

/// Capture the first and last [`BODY_SNIPPET_BYTES`] bytes of a body string.
///
/// Byte-counted (not char-counted): the snippets feed framing-anomaly
/// telemetry where byte sizes matter. A multi-byte UTF-8 sequence split at the
/// boundary decodes as a replacement character in the lossy conversion.
#[must_use]
pub(crate) fn body_head_tail(body: &str) -> (String, String) {
    let bytes = body.as_bytes();
    let head_len = bytes.len().min(BODY_SNIPPET_BYTES);
    let head = String::from_utf8_lossy(&bytes[..head_len]).into_owned();
    let tail_start = bytes.len().saturating_sub(BODY_SNIPPET_BYTES);
    let tail = String::from_utf8_lossy(&bytes[tail_start..]).into_owned();
    (head, tail)
}

/// Persist a failure record to the logs store's `retry_failures` table
/// (best-effort).
///
/// Never fails the operation — persistence failures are logged at debug.
pub(crate) async fn record_retry_failure(record: &RetryFailureRecord) {
    let Some(store) = crate::logs::LOG_STORE.get() else {
        return;
    };
    if let Err(e) = store.record_retry_failure(record).await {
        tracing::debug!(
            error = %e,
            class = record.class.label(),
            "Failed to persist retry failure record",
        );
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

    #[must_use]
    pub(crate) fn wall_clock(failures: Vec<RetryFailureRecord>) -> Self {
        Self::new(failures, FailureClass::WallClockExceeded)
    }
}

impl fmt::Display for RetryExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.detail)
    }
}

impl std::error::Error for RetryExhausted {}

// ── Shared retry-loop state (schedule + failure trail) ───────────────────

/// Mutable per-operation state shared by the outer retry loops.
///
/// Encapsulates the backoff schedule, the per-attempt failure trail,
/// Retry-After stickiness, and the wall-clock deadline so [`retry_chat`] and
/// [`crate::extraction::retry_extract_structured_scoped`] cannot drift —
/// schedule, sleep bounds, and trail mechanics live in exactly one place.
pub(crate) struct RetryLoop {
    policy: RetryPolicy,
    deadline: Instant,
    backoffs: Vec<u64>,
    failures: Vec<RetryFailureRecord>,
    last_retry_after: Option<u64>,
}

impl RetryLoop {
    /// Start a new operation: snapshots the schedule and computes the
    /// authoritative wall-clock deadline.
    #[must_use]
    pub(crate) fn new(policy: &RetryPolicy) -> Self {
        Self {
            policy: policy.clone(),
            deadline: Instant::now() + policy.operation_timeout,
            backoffs: backoff_sequence(policy),
            failures: Vec::new(),
            last_retry_after: None,
        }
    }

    /// Absolute operation deadline passed to `chat_scoped` (per-attempt total
    /// = remaining budget).
    #[must_use]
    pub(crate) fn deadline(&self) -> Instant {
        self.deadline
    }

    /// True when the operation wall-clock deadline has passed — the
    /// authoritative cap, checked before each attempt.
    #[must_use]
    pub(crate) fn expired(&self) -> bool {
        Instant::now() >= self.deadline
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

    /// Persist and record a failed attempt, stamping the attempt number and
    /// updating the sticky Retry-After. A record without a Retry-After
    /// (parse / NoResponse failures) clears any stale value from an earlier
    /// attempt, so a 429 Retry-After never bleeds into later sleeps.
    pub(crate) async fn record(&mut self, attempt: u32, mut rec: RetryFailureRecord) {
        rec.attempt = attempt;
        self.last_retry_after = rec.retry_after_ms;
        record_retry_failure(&rec).await;
        self.failures.push(rec);
    }

    /// Sleep between attempts, honoring the schedule / Retry-After and
    /// reaching the operation deadline when the schedule would exceed it —
    /// the wall cap cannot be overshot by a backoff. Returns
    /// `Err(FailureClass::Shutdown)` when the global shutdown token fires
    /// during the sleep.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) async fn sleep_between(&self, attempt: u32) -> Result<(), FailureClass> {
        if attempt >= self.policy.max_attempts {
            return Ok(());
        }
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        let schedule_ms = self.backoffs[(attempt - 1) as usize];
        // Round the remaining budget up so the sleep reaches the deadline —
        // `as_millis()` truncation would undersleep by up to 1 ms, letting
        // extra attempts run before the next `expired()` check binds.
        let remaining_ms = remaining.as_millis() as u64 + u64::from(remaining.subsec_nanos() > 0);
        let sleep_ms = compute_sleep_ms(schedule_ms, self.last_retry_after).min(remaining_ms);
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

// ── The consolidation retry loop ─────────────────────────────────────────

/// Run a chat call with the outer retry loop (used by analyst consolidation).
///
/// The request is byte-identical across ALL attempts — there is no re-prompt
/// here (consolidation has no target schema). Retryable conditions: provider
/// failures of any retryable class AND empty response text.
pub(crate) async fn retry_chat(
    request: ChatRequest,
    policy: &RetryPolicy,
) -> Result<ChatResponse, RetryExhausted> {
    retry_chat_with(request, policy, |r| {
        r.text.as_deref().is_some_and(|t| !t.trim().is_empty())
    })
    .await
}

/// Agent-loop LLM call with the same outer retry loop.
///
/// Byte-identical request across ALL attempts; retries provider failures of
/// any retryable class and honors the operation wall-clock cap. Any `Ok`
/// response is accepted — empty text is a valid tool-call turn, so unlike
/// consolidation there is no text contract.
pub(crate) async fn agent_chat(
    request: ChatRequest,
    policy: &RetryPolicy,
) -> Result<ChatResponse, RetryExhausted> {
    retry_chat_with(request, policy, |_| true).await
}

/// Shared outer retry loop: at most `policy.max_attempts` single-attempt
/// scoped calls, wall-clock-bounded, with per-attempt telemetry. `is_success`
/// decides whether an `Ok` response ends the loop (empty-text handling differs
/// between the agent path and consolidation).
#[allow(clippy::cast_possible_truncation)]
async fn retry_chat_with(
    request: ChatRequest,
    policy: &RetryPolicy,
    is_success: impl Fn(&ChatResponse) -> bool + Send,
) -> Result<ChatResponse, RetryExhausted> {
    let mut loop_state = RetryLoop::new(policy);
    let operation_started = Instant::now();

    for attempt in 1..=policy.max_attempts {
        if loop_state.expired() {
            let exhausted = RetryExhausted::wall_clock(loop_state.into_failures());
            crate::stats::record_llm_failure(&request, operation_started, &exhausted).await;
            return Err(exhausted);
        }

        let attempt_started = Instant::now();
        match crate::providers::chat_scoped(
            request.clone(),
            policy.idle_timeout,
            loop_state.deadline(),
        )
        .await
        {
            Ok(resp) => {
                if is_success(&resp) {
                    crate::stats::record_llm_success(&request, operation_started, attempt, &resp)
                        .await;
                    return Ok(resp);
                }
                let elapsed = attempt_started.elapsed().as_millis() as u64;
                let rec = RetryFailureRecord::new_simple(
                    attempt,
                    FailureClass::NoResponse,
                    &anyhow::anyhow!("LLM returned empty response"),
                    elapsed,
                    None,
                );
                loop_state.record(attempt, rec).await;
            }
            Err(err) => {
                let non_retryable = !err.class.is_retryable();
                loop_state.record(attempt, err.record).await;
                if non_retryable {
                    let exhausted = RetryExhausted::new(loop_state.into_failures(), err.class);
                    crate::stats::record_llm_failure(&request, operation_started, &exhausted).await;
                    return Err(exhausted);
                }
            }
        }

        if let Err(FailureClass::Shutdown) = loop_state.sleep_between(attempt).await {
            let exhausted = RetryExhausted::shutdown(loop_state.into_failures());
            crate::stats::record_llm_failure(&request, operation_started, &exhausted).await;
            return Err(exhausted);
        }
    }

    // Exhausted — report the last failure's class.
    let final_class = loop_state.final_class();
    let exhausted = RetryExhausted::new(loop_state.into_failures(), final_class);
    crate::stats::record_llm_failure(&request, operation_started, &exhausted).await;
    Err(exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_defaults_match_typed_defaults() {
        // The `_STR` constants feed config.rs's `string_config_fields!`
        // macro (`or(...)` annotations); the typed constants drive the
        // schedule. Keep them in lockstep.
        assert_eq!(
            DEFAULT_RETRY_MAX_ATTEMPTS_STR
                .parse::<u32>()
                .expect("max attempts str"),
            DEFAULT_RETRY_MAX_ATTEMPTS
        );
        assert_eq!(
            DEFAULT_RETRY_BASE_BACKOFF_MS_STR
                .parse::<u64>()
                .expect("base backoff str"),
            DEFAULT_RETRY_BASE_BACKOFF_MS
        );
        assert_eq!(
            DEFAULT_RETRY_MAX_BACKOFF_MS_STR
                .parse::<u64>()
                .expect("max backoff str"),
            DEFAULT_RETRY_MAX_BACKOFF_MS
        );
        assert_eq!(
            DEFAULT_OPERATION_TIMEOUT_SECS_STR
                .parse::<u64>()
                .expect("timeout secs str"),
            DEFAULT_OPERATION_TIMEOUT.as_secs()
        );
    }

    #[test]
    fn default_backoff_sequence_is_doubling_capped() {
        let p = RetryPolicy {
            max_attempts: DEFAULT_RETRY_MAX_ATTEMPTS,
            base_backoff_ms: DEFAULT_RETRY_BASE_BACKOFF_MS,
            max_backoff_ms: DEFAULT_RETRY_MAX_BACKOFF_MS,
            operation_timeout: DEFAULT_OPERATION_TIMEOUT,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
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
            operation_timeout: DEFAULT_OPERATION_TIMEOUT,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
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

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn stale_retry_after_does_not_stick_across_failures() {
        // A 429 Retry-After applies only to the sleep following the 429. A
        // later non-429 failure (parse / NoResponse, no Retry-After) must
        // clear it — otherwise a stale value wastes up to 60 s of the 720 s
        // wall budget per occurrence (reviewer finding).
        let _guard = crate::util::test::retry_tests_lock();
        let policy = tiny_test_policy();
        let mut loop_state = RetryLoop::new(&policy);

        // 429-style failure carries a Retry-After.
        let rec = RetryFailureRecord::new_simple(
            1,
            FailureClass::Transport,
            &anyhow::anyhow!("429 rate limited"),
            1,
            Some(60_000),
        );
        loop_state.record(1, rec).await;
        assert_eq!(loop_state.last_retry_after, Some(60_000));

        // A later parse/NoResponse failure has no Retry-After → clears it.
        let rec = RetryFailureRecord::new_simple(
            2,
            FailureClass::NoResponse,
            &anyhow::anyhow!("empty response"),
            1,
            None,
        );
        loop_state.record(2, rec).await;
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
        assert!(!FailureClass::WallClockExceeded.is_retryable());
        assert_eq!(
            FailureClass::TruncatedEnvelope.label(),
            "truncated_envelope"
        );
    }

    #[test]
    fn body_head_tail_splits_short_and_long() {
        assert_eq!(
            body_head_tail("short"),
            ("short".to_string(), "short".to_string())
        );
        let long = "x".repeat(1_000);
        let (head, tail) = body_head_tail(&long);
        assert_eq!(head.len(), 200);
        assert_eq!(tail.len(), 200);
        assert_eq!(head, "x".repeat(200));
        assert_eq!(tail, "x".repeat(200));
    }

    #[test]
    fn invalid_config_values_fall_back_to_defaults() {
        let _guard = crate::util::test::retry_tests_lock();
        // Corrupt every retry tunable — from_config must fall back to defaults.
        for key in [
            "retry_max_attempts",
            "retry_base_backoff_ms",
            "retry_max_backoff_ms",
            "operation_timeout_secs",
        ] {
            assert!(
                crate::config::CONFIG.set_string_field(key, "garbage"),
                "{key}"
            );
        }
        let policy = RetryPolicy::from_config();
        assert_eq!(policy.max_attempts, DEFAULT_RETRY_MAX_ATTEMPTS);
        assert_eq!(policy.base_backoff_ms, DEFAULT_RETRY_BASE_BACKOFF_MS);
        assert_eq!(policy.max_backoff_ms, DEFAULT_RETRY_MAX_BACKOFF_MS);
        assert_eq!(policy.operation_timeout, DEFAULT_OPERATION_TIMEOUT);

        // Zero / negative are also invalid (parse_cfg_u64 rejects v == 0).
        for key in [
            "retry_max_attempts",
            "retry_base_backoff_ms",
            "retry_max_backoff_ms",
            "operation_timeout_secs",
        ] {
            assert!(crate::config::CONFIG.set_string_field(key, "0"), "{key}");
        }
        let policy = RetryPolicy::from_config();
        assert_eq!(policy.max_attempts, DEFAULT_RETRY_MAX_ATTEMPTS);

        // Restore the global for other tests.
        for key in [
            "retry_max_attempts",
            "retry_base_backoff_ms",
            "retry_max_backoff_ms",
            "operation_timeout_secs",
        ] {
            let _ = crate::config::CONFIG.set_string_field(key, "");
        }
    }

    #[test]
    fn custom_config_values_are_respected() {
        let _guard = crate::util::test::retry_tests_lock();
        assert!(crate::config::CONFIG.set_string_field("retry_max_attempts", "11"));
        assert!(crate::config::CONFIG.set_string_field("retry_base_backoff_ms", "1234"));
        assert!(crate::config::CONFIG.set_string_field("retry_max_backoff_ms", "99999"));
        assert!(crate::config::CONFIG.set_string_field("operation_timeout_secs", "120"));

        let policy = RetryPolicy::from_config();
        assert_eq!(policy.max_attempts, 11);
        assert_eq!(policy.base_backoff_ms, 1_234);
        assert_eq!(policy.max_backoff_ms, 99_999);
        assert_eq!(policy.operation_timeout, Duration::from_mins(2));

        for key in [
            "retry_max_attempts",
            "retry_base_backoff_ms",
            "retry_max_backoff_ms",
            "operation_timeout_secs",
        ] {
            let _ = crate::config::CONFIG.set_string_field(key, "");
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn agent_chat_rides_out_sustained_outage_and_recovers() {
        let _guard = crate::util::test::retry_tests_lock();
        // The binding agent-loop budget (13 attempts): the first 12 attempts
        // hit a sustained 503-style outage, the 13th recovers.
        let policy = RetryPolicy {
            max_attempts: 13,
            base_backoff_ms: 1,
            max_backoff_ms: 1,
            operation_timeout: Duration::from_mins(12),
            idle_timeout: Duration::from_secs(1),
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

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn agent_chat_wall_clock_cap_binds() {
        let _guard = crate::util::test::retry_tests_lock();
        // 13 attempts but a 100 ms wall cap and 1 s backoff — the cap binds.
        let policy = RetryPolicy {
            max_attempts: 13,
            base_backoff_ms: 1_000,
            max_backoff_ms: 1_000,
            operation_timeout: Duration::from_millis(100),
            idle_timeout: Duration::from_secs(1),
        };
        let fake = crate::util::test::FakeProvider::new()
            .err(FailureClass::Transport, "slow outage")
            .err(FailureClass::Transport, "slow outage");
        let _provider_guard = crate::util::test::install_fake_provider(std::sync::Arc::new(fake));
        let request = crate::providers::test_request(vec![crate::ChatMessage::user("hi")], None);
        let failure = agent_chat(request, &policy)
            .await
            .expect_err("wall-clock cap must bind");
        assert_eq!(failure.final_class, FailureClass::WallClockExceeded);
        assert!(
            failure.failures.len() <= 2,
            "cap must stop the loop before 13 attempts"
        );
    }
}
