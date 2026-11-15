//! TTL-ladder executor for the OpenRouter provider benchmark.
//!
//! This is a STANDALONE bench HTTP path — it deliberately does NOT use
//! [`crate::providers::compatible`], [`crate::retry`] or the global CONFIG provider stack. It
//! owns its reqwest client (provided by the full-run orchestrator), its own bounded
//! non-jittered retry loop and its own spend accounting, so the measurement is independent of
//! the production agent pipeline.
//!
//! # Protocol
//!
//! - **Canonical deterministic requests**: every body is built from the same base prompt plus
//!   the verified tool frames of prior rounds, and the prompt prefix is byte-identical across
//!   warmup and ladder rounds so provider-side prompt caches can be measured.
//! - **Warmup gate**: two byte-identical warmup requests (W1, W2) establish the base cache; W1
//!   also gates auth/quota failures before any spend.
//! - **Cache-hold predicate**: a provider "does not cache" ONLY when its raw cached tokens are
//!   zero across warmup-2 and every ladder round (warmup-1 is cross-run contaminated). Any
//!   non-zero observation means it caches, and the bucket comes from the ladder. Providers that
//!   failed to run are "not measured"; a ladder cut short before its last rung without an
//!   observed drop is reported as truncated (see [`crate::bench_openrouter::classify`]).
//! - **TTL ladder**: a list of inactivity GAPS between rounds — a gap list of N entries opens
//!   N + 1 ladder rounds ([`crate::bench_openrouter::rounds_per_provider`] adds the warmups).
//!   Each round re-sends the grown conversation after its gap and compares the reported cached
//!   tokens against the full-hold expectation. Every selected endpoint walks the WHOLE configured
//!   ladder: there is no early stop on a drop, because endpoints are measured concurrently (the
//!   run lasts as long as the slowest endpoint) and a dropped endpoint's bucket is already
//!   settled.
//! - **Bounded non-jittered retries**: at most 3 attempts per round; sleeps are fixed and NEVER
//!   jittered — jitter would corrupt the gap measurement.
//! - **Spend accounting**: per-round billed cost accumulates against a per-provider guard and a
//!   total cap ([`RunBudget`]); the guard stops this provider's ladder, the cap aborts the run.
//! - **Deadline + abort**: [`RUN_DEADLINE_MINS`] caps the run, and the per-provider deadline and
//!   the run-level abort token race every ladder sleep, so a stalled provider cannot hang the
//!   bench. A full default ladder needs well over an hour of wall clock; the orchestrator wraps
//!   the whole run in [`OUTER_TIMEOUT`] ([`OUTER_TIMEOUT_MINS`]), which is deliberately larger
//!   than the deadline: an in-flight request is not raced against the deadline and can overshoot
//!   it, so a clean truncation must not become a killed run.
//! - **Pinning verification**: the serving provider is checked against the endpoint's names; a
//!   drift is retried once and then marked Invalid (`"pin_drift"`).
//! - **Response-cache detection**: a `hit` in `x-openrouter-cache-status` means OpenRouter
//!   answered from its response cache without exercising the model — re-sent once after a short
//!   delay, and marked Invalid (`"response_cache"`) if it persists.
//! - **Stop lines**: for an endpoint whose ladder did NOT run to its end, the run log carries one
//!   `bench [tag] stopped: <reason>` line ([`StopReason`]), written by that endpoint's own task
//!   ([`StopLine`]) at the moment it stops. The condition is "did not complete its ladder", not
//!   the truncation flag: a final-rung guard or cap trip is a complete ladder and gets no line.
//!
//! `run_provider` is driven by the full-run scheduler in [`crate::bench_openrouter`] and returns
//! a [`ProviderRun`] carrying the tag and the derived cache-hold result.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::bench_openrouter::classify::{
    CacheClassification, cache_hold_result, classify_round, expected_cached_for_round,
    is_truncated, truncated_result, verify_pinned,
};
use crate::bench_openrouter::discovery::EndpointInfo;
use crate::util::{UnwrapPoison, print_stderr};

/// Whole-run deadline (minutes). The orchestrator derives the per-provider deadline from the
/// run start plus this constant and stores it in [`RunContext::deadline`]; the executor races
/// it and never computes it. A full default ladder ([`DEFAULT_LADDER_SECS`]) spends the sum of
/// its gaps asleep, plus every round's request overhead on top, so the deadline has to leave
/// real room above that sum — the budget test pins the floor.
pub(crate) const RUN_DEADLINE_MINS: u64 = 130;

/// Outer timeout (minutes) for the whole run, applied by the orchestrator around the provider
/// tasks. Declared independently of [`RUN_DEADLINE_MINS`] on purpose: the margin is the
/// headroom for a request that is already in flight when the deadline fires — in-flight
/// requests are not raced against the deadline and may overshoot it by their own timeout, and
/// a clean truncation must not turn into a killed run.
pub(crate) const OUTER_TIMEOUT_MINS: u64 = 160;

/// [`OUTER_TIMEOUT_MINS`] as the orchestrator's timeout value.
pub(crate) const OUTER_TIMEOUT: std::time::Duration =
    std::time::Duration::from_mins(OUTER_TIMEOUT_MINS);

/// Default inactivity-gap ladder (seconds). The top rung is one hour, so an endpoint that keeps
/// a prompt cache for a full hour is distinguishable from one that holds half an hour; the gaps
/// are the waits BETWEEN rounds, and [`crate::bench_openrouter::rounds_per_provider`] turns the
/// list into the request count. `BenchOptions::parse` takes its default from here.
pub(crate) const DEFAULT_LADDER_SECS: &[u64] = &[0, 5, 30, 120, 300, 600, 1800, 3600];

const CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

const REQUEST_TIMEOUT_SECS: u64 = 120;

const MAX_ATTEMPTS: u32 = 3;

/// Retry sleep floor and ceiling (ms) — Retry-After is clamped into this range; other
/// retryable statuses sleep the floor. Never jittered.
const RETRY_SLEEP_MIN_MS: u64 = 5_000;
const RETRY_SLEEP_MAX_MS: u64 = 60_000;

const RESPONSE_CACHE_RETRY_DELAY_SECS: u64 = 2;

const ROUND_MAX_TOKENS: u32 = 128;

const LENGTH_RETRY_MAX_TOKENS: u32 = 1024;

const FILLER_WORDS: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa",
];

/// The base prompt for one provider run: the harness system prompt plus the user message,
/// byte-identical across the run's rounds so the prefix is cacheable.
#[derive(Clone)]
pub(crate) struct BasePrompt {
    pub system: String,
    pub user: String,
}

/// Build the base prompt from the harness system prompt, a per-run nonce and the deterministic
/// filler blob. The nonce is in the user message so different runs are distinguishable in the
/// API logs while remaining byte-identical within one run; the message also carries the fixed
/// step-0 instruction (later steps come from the tool results).
#[must_use]
pub(crate) fn build_base_prompt(system: &str, nonce: &str, filler: &str) -> BasePrompt {
    BasePrompt {
        system: system.to_string(),
        user: format!(
            "{filler}\n\nStep 0: call fast_tool with step 0 and nothing else.\n[bench-run {nonce}]"
        ),
    }
}

/// Deterministic filler blob: numbered lines from [`FILLER_WORDS`] joined with '\n', stopping
/// once the length reaches `prefix_chars` (a partial final line is fine). No randomness and no
/// wall clock, so two calls with the same argument are byte-identical.
#[must_use]
pub(crate) fn generate_filler(prefix_chars: usize) -> String {
    let mut out = String::new();
    let mut i = 0usize;
    while out.len() < prefix_chars {
        if i > 0 {
            out.push('\n');
        }
        let _ = write!(out, "{:06} {}", i, FILLER_WORDS[i % FILLER_WORDS.len()]);
        i += 1;
    }
    out
}

/// One verified tool-call frame: the model called `fast_tool` with a step, and the executor
/// records the call id and any reasoning fields for the next round's history.
struct ToolFrame {
    pub id: String,
    pub step: u64,
    pub reasoning: Option<serde_json::Value>,
}

/// Build the message array for a round: `[system, user]` then one assistant(tool_calls) +
/// tool(result) pair per verified frame, with a frame's `reasoning` keys spread into the
/// assistant message. The step the round expects lives in the last tool result, NOT in the
/// prompt prefix — that would change it per round and break the cache measurement.
#[must_use]
fn build_messages(base: &BasePrompt, frames: &[ToolFrame]) -> Vec<serde_json::Value> {
    let mut messages = vec![
        json!({"role": "system", "content": base.system}),
        json!({"role": "user", "content": base.user}),
    ];
    for frame in frames {
        let mut assistant = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": frame.id,
                "type": "function",
                "function": {
                    "name": "fast_tool",
                    "arguments": format!("{{\"step\":{}}}", frame.step),
                },
            }],
        });
        if let Some(reasoning) = &frame.reasoning
            && let Some(obj) = reasoning.as_object()
        {
            for (key, value) in obj {
                assistant[key] = value.clone();
            }
        }
        messages.push(assistant);
        messages.push(json!({
            "role": "tool",
            "tool_call_id": frame.id,
            "content": format!(
                "step {} acknowledged; proceed to step {}",
                frame.step,
                frame.step + 1
            ),
        }));
    }
    messages
}

/// The canonical request body for one round. `provider.order` pins the endpoint's tag slug with
/// fallbacks off; `tool_choice` and a `None` `reasoning_effort` are omitted so byte-identical
/// requests stay byte-identical. The expected step is not in the body either — it lives in the
/// tool results.
#[must_use]
fn build_request_body(
    model: &str,
    base: &BasePrompt,
    frames: &[ToolFrame],
    tag: &str,
    reasoning_effort: Option<&str>,
    max_tokens: u32,
) -> serde_json::Value {
    let mut body = json!({
        "model": model,
        "messages": build_messages(base, frames),
        "max_tokens": max_tokens,
        "tools": [{
            "type": "function",
            "function": {
                "name": "fast_tool",
                "description": "Report the current benchmark step.",
                "parameters": {
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                    "required": ["step"],
                },
            },
        }],
        "provider": {"order": [tag], "allow_fallbacks": false},
    });
    if let Some(effort) = reasoning_effort {
        body["reasoning_effort"] = json!(effort);
    }
    body
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// Raw `usage` object, trimmed to what the run consumes (prompt, cached, cost); serde ignores
/// the other wire fields.
#[derive(Debug, Clone, Default, Deserialize)]
struct RawUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    #[serde(default)]
    pub cost: Option<f64>,
}

/// Raw chat-completions response envelope. Every field is optional, so a missing or malformed one
/// is a parse failure (recorded, not fatal); serde ignores the wire fields the run does not read.
#[derive(Debug, Clone, Default, Deserialize)]
struct RawEnvelope {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub usage: Option<RawUsage>,
    #[serde(default)]
    pub choices: Vec<RawChoice>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawChoice {
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub message: RawMessage,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawMessage {
    #[serde(default)]
    pub tool_calls: Option<Vec<RawToolCall>>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<RawToolCallFunction>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawToolCallFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Result of one HTTP round (one [`send_round`] call, including its retries), exposing only
/// what the ladder consumes: the envelope, the error class, the response-cache flag and the
/// extracted tool-call fields.
struct RoundOutcome {
    pub envelope: Option<RawEnvelope>,
    pub error_class: Option<String>,
    pub response_cache_hit: bool,
    pub tool_call_step: Option<u64>,
    pub tool_call_id: Option<String>,
    pub reasoning: Option<serde_json::Value>,
}

impl RoundOutcome {
    /// The invalid-round reason for a round without an envelope: the recorded error class, or
    /// `"http_error"` when none was recorded.
    fn error_reason(&self) -> String {
        self.error_class
            .as_deref()
            .unwrap_or("http_error")
            .to_string()
    }
}

/// Sleep (ms) before retrying: 429 honors Retry-After clamped into [5000, 60000] (5000 when
/// absent); any other retryable status sleeps a fixed 5000. Never jittered — jitter would
/// corrupt the TTL gap measurement.
#[must_use]
fn retry_sleep_ms(status: reqwest::StatusCode, retry_after: Option<u64>) -> u64 {
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        retry_after
            .unwrap_or(RETRY_SLEEP_MIN_MS)
            .clamp(RETRY_SLEEP_MIN_MS, RETRY_SLEEP_MAX_MS)
    } else {
        RETRY_SLEEP_MIN_MS
    }
}

/// Map a non-2xx status to its error class; retryable statuses (429, 5xx) return `None` so the
/// caller retries.
#[must_use]
fn classify_http_error(status: reqwest::StatusCode) -> Option<&'static str> {
    match status.as_u16() {
        401 => Some("auth"),
        402 => Some("quota"),
        429 | 500..=599 => None,
        _ => Some("http_4xx"),
    }
}

/// Send one round's request with bounded, never-jittered retries and return the outcome. Every
/// failure is recorded inside the [`RoundOutcome`] (never an `Err`): auth/quota/other-4xx set
/// `error_class` without retrying, retryable statuses and transport/timeout errors are retried,
/// and a final failure still yields an outcome with the class set. The API key and body are
/// never logged; at most a `tracing::debug!` per attempt.
///
/// Retry policy (3 attempts): 429 → Retry-After clamped into [5 s, 60 s], retry; 5xx → 5 s,
/// retry; transport/timeout → 5 s, retry; 401 → `"auth"`, 402 → `"quota"`, other 4xx →
/// `"http_4xx"`, no retry; 2xx → inspect `x-openrouter-cache-status` and parse (a parse failure
/// → `"parse"`).
async fn send_round(client: &reqwest::Client, key: &str, body: &serde_json::Value) -> RoundOutcome {
    let mut attempts = 0u32;
    let mut http_status = 0u16;
    let mut envelope: Option<RawEnvelope> = None;
    let mut error_class: Option<&'static str> = None;
    let mut response_cache_hit = false;

    while attempts < MAX_ATTEMPTS {
        attempts += 1;
        tracing::debug!(attempt = attempts, "bench round send");

        let resp = client
            .post(CHAT_COMPLETIONS_URL)
            .bearer_auth(key)
            .json(body)
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                error_class = Some(if e.is_timeout() {
                    "timeout"
                } else {
                    "transport"
                });
                if attempts < MAX_ATTEMPTS {
                    tokio::time::sleep(Duration::from_millis(RETRY_SLEEP_MIN_MS)).await;
                }
                continue;
            }
        };

        let status = resp.status();
        http_status = status.as_u16();
        if status.is_success() {
            let cache_status = resp
                .headers()
                .get("x-openrouter-cache-status")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            response_cache_hit = cache_status
                .as_deref()
                .is_some_and(|s| s.to_ascii_lowercase().contains("hit"));
            match resp.json::<RawEnvelope>().await {
                Ok(env) => envelope = Some(env),
                Err(_) => error_class = Some("parse"),
            }
            break;
        }

        if let Some(class) = classify_http_error(status) {
            error_class = Some(class);
            break; // auth / quota / other 4xx: no retry
        }
        let retry_after = crate::util::error::retry_after_header(resp.headers());
        if attempts < MAX_ATTEMPTS {
            tokio::time::sleep(Duration::from_millis(retry_sleep_ms(status, retry_after))).await;
        }
    }

    // Exhausted retryable statuses leave error_class None — record a concrete class so the
    // caller can classify the round as failed.
    if error_class.is_none() && envelope.is_none() && http_status != 0 {
        error_class = Some(if http_status == 429 {
            "rate_limited"
        } else {
            "http_5xx"
        });
    }

    let (tool_call_step, tool_call_id, reasoning) = extract_tool_call(envelope.as_ref());

    RoundOutcome {
        envelope,
        error_class: error_class.map(str::to_string),
        response_cache_hit,
        tool_call_step,
        tool_call_id,
        reasoning,
    }
}

/// The first `fast_tool` tool call of the envelope: its parsed `step` argument, its call id,
/// and a combined reasoning object from the first choice's message.
#[must_use]
fn extract_tool_call(
    envelope: Option<&RawEnvelope>,
) -> (Option<u64>, Option<String>, Option<serde_json::Value>) {
    let Some(choice) = envelope.and_then(|e| e.choices.first()) else {
        return (None, None, None);
    };
    let Some(calls) = &choice.message.tool_calls else {
        return (None, None, None);
    };
    let Some(call) = calls.iter().find(|c| {
        c.function
            .as_ref()
            .is_some_and(|f| f.name.as_deref() == Some("fast_tool"))
    }) else {
        return (None, None, None);
    };
    let step = call
        .function
        .as_ref()
        .and_then(|f| f.arguments.as_deref())
        .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
        .and_then(|v| v.get("step").and_then(serde_json::Value::as_u64));
    (step, call.id.clone(), combine_reasoning(&choice.message))
}

/// Combine a message's reasoning fields into one object when any are present.
#[must_use]
fn combine_reasoning(msg: &RawMessage) -> Option<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    if let Some(v) = &msg.reasoning_content {
        obj.insert("reasoning_content".to_string(), json!(v));
    }
    if let Some(v) = &msg.reasoning {
        obj.insert("reasoning".to_string(), json!(v));
    }
    if let Some(v) = &msg.reasoning_details {
        obj.insert("reasoning_details".to_string(), v.clone());
    }
    if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj))
    }
}

/// Aggregated usage of one round — only the fields the cache-hold predicate and the budget need.
#[derive(Debug, Clone, Default)]
struct RoundUsage {
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub cost: Option<f64>,
}

/// Cached tokens from a raw usage object: `prompt_tokens_details.cached_tokens`, falling back
/// to the Anthropic-style `prompt_cache_hit_tokens`.
#[must_use]
fn raw_cached(u: &RawUsage) -> Option<u64> {
    u.prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .or(u.prompt_cache_hit_tokens)
}

#[must_use]
fn cached_tokens_of(envelope: Option<&RawEnvelope>) -> u64 {
    envelope
        .and_then(|e| e.usage.as_ref())
        .and_then(raw_cached)
        .unwrap_or(0)
}

#[must_use]
fn usage_from(envelope: Option<&RawEnvelope>) -> RoundUsage {
    let Some(u) = envelope.and_then(|e| e.usage.as_ref()) else {
        return RoundUsage::default();
    };
    RoundUsage {
        prompt_tokens: u.prompt_tokens,
        cached_tokens: raw_cached(u),
        cost: u.cost,
    }
}

pub(crate) struct RunBudget {
    pub cap_usd: f64,
    pub per_provider_guard: f64,
    pub total_spent: std::sync::Mutex<f64>,
    pub provider_spent: std::sync::Mutex<HashMap<String, f64>>,
}

impl RunBudget {
    #[must_use]
    pub(crate) fn new(cap_usd: f64, per_provider_guard: f64) -> Self {
        Self {
            cap_usd,
            per_provider_guard,
            total_spent: std::sync::Mutex::new(0.0),
            provider_spent: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Record `billed` USD against `tag`, answering `(over_guard, over_cap)` — true when the
    /// provider's guard or the total cap is exceeded AFTER this record.
    #[must_use]
    pub(crate) fn record(&self, tag: &str, billed: f64) -> (bool, bool) {
        let mut total = self.total_spent.lock().unwrap_poison();
        *total += billed;
        let mut providers = self.provider_spent.lock().unwrap_poison();
        let spent = providers.entry(tag.to_string()).or_insert(0.0);
        *spent += billed;
        (*spent > self.per_provider_guard, *total > self.cap_usd)
    }

    #[must_use]
    pub(crate) fn total(&self) -> f64 {
        *self.total_spent.lock().unwrap_poison()
    }

    fn cap_exceeded_reason(&self) -> String {
        format!(
            "spend cap exceeded (${:.4} > ${:.4})",
            self.total(),
            self.cap_usd
        )
    }
}

/// Why an endpoint's ladder stopped before its last rung.
#[derive(Clone, Debug, PartialEq, Eq)]
enum StopReason {
    /// A warmup round failed: the round's own error class — `transport`/`timeout` for a request
    /// that never landed, `http_4xx`/`http_5xx`/`rate_limited`/`parse` for an HTTP or body
    /// failure, `auth`/`quota` for a refusal. Naming the class is what keeps a transport failure
    /// and a timeout, or a refusal and an HTTP error, from reading alike.
    WarmupFailed(String),
    /// This endpoint's own spend crossed the per-provider guard: its ladder stops, the run goes
    /// on.
    SpendGuardTripped,
    /// The total cap was crossed: this endpoint's ladder stops and the run aborts.
    SpendCapTripped,
    /// [`RUN_DEADLINE_MINS`] fired.
    RunDeadlineReached,
    /// The run-level abort token fired — another endpoint's cap trip or refusal, or the outer
    /// timeout. Names the cause class, never the abort's recorded reason.
    RunAborted(AbortCause),
    /// The endpoint's task died (a panic or a cancelled task) without a result.
    TaskDied,
}

impl StopReason {
    /// The stop line: the same stream and the same `bench [tag]` prefix as the per-rung lines,
    /// with a shape of its own (`stopped: …`, never `rung …`) so a reader that counts rung lines
    /// is unaffected.
    #[must_use]
    fn line(&self, tag: &str) -> String {
        format!("bench [{tag}] stopped: {}", self.describe())
    }

    /// Plain and factual — the class of the stop, never a verdict on the endpoint.
    #[must_use]
    fn describe(&self) -> String {
        match self {
            Self::WarmupFailed(class) => format!("warmup failed ({class})"),
            Self::SpendGuardTripped => "per-provider spend guard tripped".to_string(),
            Self::SpendCapTripped => "total spend cap tripped".to_string(),
            Self::RunDeadlineReached => "run deadline reached".to_string(),
            Self::RunAborted(cause) => format!("run aborted ({})", cause.as_str()),
            Self::TaskDied => "task died without a result".to_string(),
        }
    }
}

/// The class of a run-level abort — what a stop line names in place of the abort's recorded
/// reason, which carries a dollar amount and the aborting endpoint's tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AbortCause {
    SpendCap,
    AuthRefused,
    QuotaRefused,
    OuterTimeout,
}

impl AbortCause {
    /// The cause of a warmup refusal, `None` for every other class: a transport, timeout, HTTP or
    /// parse failure gates only that endpoint's ladder and never aborts the run.
    #[must_use]
    fn refusal(class: &str) -> Option<Self> {
        match class {
            "auth" => Some(Self::AuthRefused),
            "quota" => Some(Self::QuotaRefused),
            _ => None,
        }
    }

    #[must_use]
    fn as_str(self) -> &'static str {
        match self {
            Self::SpendCap => "total spend cap",
            Self::AuthRefused => "auth refusal",
            Self::QuotaRefused => "quota refusal",
            Self::OuterTimeout => "outer timeout",
        }
    }
}

pub(crate) struct RunContext {
    budget: RunBudget,
    deadline: Instant,
    abort: CancellationToken,
    /// The FIRST run-level abort: one record holding the cause class (for the stop lines) and the
    /// human reason (for the report), so the two cannot disagree.
    abort_record: std::sync::Mutex<Option<AbortRecord>>,
}

struct AbortRecord {
    cause: AbortCause,
    reason: String,
}

impl RunContext {
    #[must_use]
    pub(crate) fn new(budget: RunBudget, deadline: Instant) -> Self {
        Self {
            budget,
            deadline,
            abort: CancellationToken::new(),
            abort_record: std::sync::Mutex::new(None),
        }
    }

    /// Abort the whole run: record `cause` and `reason`, then cancel the token every ladder sleep
    /// races. The record goes first, so a ladder that wakes on the token always reads the cause
    /// it names; the first cause wins, so a provider that crosses the cap and then fails its
    /// warmup cannot rename the run's cause.
    pub(crate) fn abort(&self, cause: AbortCause, reason: String) {
        self.abort_record
            .lock()
            .unwrap_poison()
            .get_or_insert(AbortRecord { cause, reason });
        self.abort.cancel();
    }

    #[must_use]
    fn is_aborted(&self) -> bool {
        self.abort.is_cancelled()
    }

    /// Resolves when the run-level abort fires.
    async fn aborted(&self) {
        self.abort.cancelled().await;
    }

    /// The abort's human reason, for the report — never for a stop line.
    #[must_use]
    pub(crate) fn abort_reason(&self) -> Option<String> {
        self.abort_record
            .lock()
            .unwrap_poison()
            .as_ref()
            .map(|r| r.reason.clone())
    }

    /// The stop reason for a ladder cut by the run-level abort: the cause class only.
    #[must_use]
    fn abort_stop(&self) -> StopReason {
        // `abort` records the cause before cancelling the token and is the only canceller.
        let cause = self
            .abort_record
            .lock()
            .unwrap_poison()
            .as_ref()
            .map(|r| r.cause)
            .expect("an abort records its cause before cancelling the token");
        StopReason::RunAborted(cause)
    }
}

pub(crate) struct ProviderRun {
    pub tag: String,
    pub cache_hold_bucket: String,
    /// True iff the ladder ended before its last rung without an observed drop, i.e. exactly
    /// when `cache_hold_bucket` carries the classify module's truncation suffix — both come from
    /// [`is_truncated`].
    pub truncated: bool,
}

impl ProviderRun {
    /// A provider that never produced a usable response (warmup or transport failure): `not
    /// measured`, and not a truncated row — its ladder was never going to run.
    #[must_use]
    pub(crate) fn not_measured(tag: String) -> Self {
        Self {
            tag,
            cache_hold_bucket: "not measured".to_string(),
            truncated: false,
        }
    }

    /// A provider whose warmups succeeded but whose ladder never ran (the warmup alone tripped the
    /// spend guard or the total cap): the ladder ended before its first rung with no observed drop,
    /// so the answer could still have changed. The row reads `not measured (ladder truncated)` —
    /// never a bare `not measured`, which would claim the endpoint produced nothing usable. Built
    /// through the same predicate and formatter as every other truncated row, so the suffix cannot
    /// drift.
    #[must_use]
    pub(crate) fn ladder_never_ran(tag: String) -> Self {
        Self {
            tag,
            cache_hold_bucket: truncated_result(&[], &[]),
            truncated: is_truncated(false, &[]),
        }
    }
}

/// Writes an endpoint's stop line when its task ends without finishing its ladder.
///
/// Armed pessimistically as [`StopReason::TaskDied`] and disarmed by [`Self::complete`], so one
/// place covers every exit — a panic, a cancelled task, an early return — and the line is written
/// by the endpoint's own task, at the moment it stops.
pub(crate) struct StopLine {
    tag: String,
    reason: Option<StopReason>,
}

impl StopLine {
    /// Arm the guard for `tag`: it writes "task died" unless [`Self::stop`] or [`Self::complete`]
    /// settles it.
    #[must_use]
    pub(crate) fn new(tag: String) -> Self {
        Self {
            tag,
            reason: Some(StopReason::TaskDied),
        }
    }

    /// Record why the endpoint stopped; written when the guard drops.
    fn stop(&mut self, reason: StopReason) {
        self.reason = Some(reason);
    }

    /// The ladder ran to its last rung: no line.
    fn complete(&mut self) {
        self.reason = None;
    }

    /// The line the guard will write, `None` once the ladder completed.
    #[must_use]
    fn pending(&self) -> Option<String> {
        self.reason.as_ref().map(|reason| reason.line(&self.tag))
    }
}

impl Drop for StopLine {
    fn drop(&mut self) {
        if let Some(line) = self.pending() {
            print_stderr(&line);
        }
    }
}

/// The one stop decision: only a ladder that did not reach its last rung reports a reason.
#[must_use]
fn stop_reason(completed: bool, stop: Option<StopReason>) -> Option<StopReason> {
    stop.filter(|_| !completed)
}

/// Run one provider's warmup + TTL ladder. Never panics on provider behavior: every failure
/// degrades to a `not measured` row (a warmup or transport failure), a truncated row (the warmup
/// alone tripped the spend guard or cap), or a ladder-derived bucket. `stop_line` is settled on
/// every path, so an early return has to name its cause.
#[expect(
    clippy::too_many_arguments,     // the full protocol inputs; bundled would obscure the call site
    clippy::too_many_lines,         // one long sequential protocol implementation
    clippy::cast_precision_loss,    // gap seconds (u64) → f64 for the nominal-gap bookkeeping
    clippy::ignored_unit_patterns   // spec'd select! arms bind the unit sleep future with _
)]
pub(crate) async fn run_provider(
    client: &reqwest::Client,
    key: &str,
    endpoint: &EndpointInfo,
    model: &str,
    base: &BasePrompt,
    reasoning_effort: Option<&str>,
    ladder_secs: &[u64],
    context: &RunContext,
    run_started: Instant,
    stop_line: &mut StopLine,
) -> ProviderRun {
    let tag = endpoint.tag.clone();

    // W1: warmup 1 — the auth/quota gate before any spend
    let w_body = build_request_body(model, base, &[], &tag, reasoning_effort, ROUND_MAX_TOKENS);
    let outcome_w1 = send_round(client, key, &w_body).await;

    // A failed W1 round gates the ladder: auth/quota abort the whole run; any
    // other error class skips this provider's ladder without touching the run
    // token (no wasted W2 + ladder spend).
    if let Some(class) = outcome_w1.error_class.as_deref() {
        stop_line.stop(StopReason::WarmupFailed(class.to_string()));
        if let Some(cause) = AbortCause::refusal(class) {
            context.abort(cause, format!("warmup failed ({class}); provider '{tag}'"));
            tracing::debug!(tag = %tag, class, "provider aborted during warmup");
        } else {
            tracing::debug!(tag = %tag, class, "provider failed warmup; skipping ladder");
        }
        return ProviderRun::not_measured(tag);
    }

    // W2: warmup 2 — byte-identical; the base-cache sample
    let outcome_w2 = send_round(client, key, &w_body).await;
    let w2_cached = cached_tokens_of(outcome_w2.envelope.as_ref());
    let w2_is_response_cache = outcome_w2.response_cache_hit;
    // A response-cache hit is not a genuine provider measurement (OpenRouter
    // answered from its response cache without running the model): its cached
    // tokens are stale. Exclude it from both the base-cache anchor and the
    // zero-cache predicate. (prompt_tokens stays trustworthy — it is derived from
    // the request, not the cache state.)
    let base_cached = if w2_is_response_cache { 0 } else { w2_cached };
    let w2_prompt_tokens = outcome_w2
        .envelope
        .as_ref()
        .and_then(|e| e.usage.as_ref())
        .and_then(|u| u.prompt_tokens);
    // The zero-cache predicate uses W2 + ladder rounds only (W1 is excluded by
    // construction since it is cross-run contaminated).
    let mut cached_observations: Vec<u64> = if w2_is_response_cache {
        Vec::new()
    } else {
        vec![w2_cached]
    };

    // Warmup spend counts against the budget: the cap covers ALL billed usage, not just ladder
    // rounds. The record — and the cap abort it can trigger — happens BEFORE the W2 gate: a run
    // that has crossed its cap must cancel the other providers' ladder sleeps whatever this
    // provider's row ends up saying.
    let w_cost = usage_from(outcome_w1.envelope.as_ref()).cost.unwrap_or(0.0)
        + usage_from(outcome_w2.envelope.as_ref()).cost.unwrap_or(0.0);
    let (over_guard, over_cap) = context.budget.record(&tag, w_cost);
    if over_cap {
        context.abort(AbortCause::SpendCap, context.budget.cap_exceeded_reason());
    }

    // W2 failure gates the ladder symmetrically with W1: auth/quota abort the whole run; any
    // other error class skips this provider's ladder (running it with base_cached=0 would
    // mis-scale every classification and burn the full ladder on likely-failing rounds). A
    // failed warmup is `not measured` whatever the budget says — its ladder never started.
    if let Some(class) = outcome_w2.error_class.as_deref() {
        stop_line.stop(StopReason::WarmupFailed(class.to_string()));
        if let Some(cause) = AbortCause::refusal(class) {
            context.abort(cause, format!("warmup failed ({class}); provider '{tag}'"));
            tracing::debug!(tag = %tag, class, "provider aborted during warmup 2");
        } else {
            tracing::debug!(tag = %tag, class, "provider failed warmup 2; skipping ladder");
        }
        return ProviderRun::not_measured(tag);
    }

    // Both warmups succeeded but their cost alone tripped the budget: the ladder ends before
    // its first rung with no observed drop, so the row is truncated, not "not measured".
    if over_cap || over_guard {
        stop_line.stop(if over_cap {
            StopReason::SpendCapTripped
        } else {
            StopReason::SpendGuardTripped
        });
        return ProviderRun::ladder_never_ran(tag);
    }

    let rounds = ladder_secs.len() + 1;
    let mut frames: Vec<ToolFrame> = Vec::new();
    let mut classifications: Vec<CacheClassification> = Vec::new();
    let mut nominal_gaps: Vec<f64> = Vec::new();
    let mut round0_prompt_tokens: Option<u64> = None;
    // Why the loop broke; `None` when it never did. Every break names its cause.
    let mut ladder_stop: Option<StopReason> = None;
    for r in 0..rounds {
        if context.is_aborted() {
            ladder_stop = Some(context.abort_stop());
            break;
        }
        if Instant::now() >= context.deadline {
            ladder_stop = Some(StopReason::RunDeadlineReached);
            break;
        }

        let next_step = u64::try_from(r).expect("ladder round index fits u64");
        let body = build_request_body(
            model,
            base,
            &frames,
            &tag,
            reasoning_effort,
            ROUND_MAX_TOKENS,
        );
        let nominal_gap_secs = if r == 0 {
            0.0
        } else {
            ladder_secs[r - 1] as f64
        };

        let mut outcome = send_round(client, key, &body).await;
        let mut round_billed = usage_from(outcome.envelope.as_ref()).cost.unwrap_or(0.0);
        let mut invalid_reason: Option<String> = None;

        // 1. Envelope present? (send_round already recorded the error class.)
        if outcome.envelope.is_none() {
            invalid_reason = Some(outcome.error_reason());
        }

        // 2. Response-cache detection: a hit means OpenRouter answered from
        //    its response cache without exercising the model — re-send once.
        if invalid_reason.is_none() && outcome.response_cache_hit {
            tokio::time::sleep(Duration::from_secs(RESPONSE_CACHE_RETRY_DELAY_SECS)).await;
            outcome = send_round(client, key, &body).await;
            round_billed += usage_from(outcome.envelope.as_ref()).cost.unwrap_or(0.0);
            if outcome.response_cache_hit {
                invalid_reason = Some("response_cache".to_string());
            } else if outcome.envelope.is_none() {
                invalid_reason = Some(outcome.error_reason());
            }
        }

        // 3. Pin verification: the serving provider must match the pinned
        //    endpoint. An absent provider is unverifiable, not drift.
        if invalid_reason.is_none() {
            let serving = outcome
                .envelope
                .as_ref()
                .and_then(|e| e.provider.as_deref());
            if serving.is_some() && !verify_pinned(serving, endpoint) {
                outcome = send_round(client, key, &body).await;
                round_billed += usage_from(outcome.envelope.as_ref()).cost.unwrap_or(0.0);
                if outcome.envelope.is_none() {
                    invalid_reason = Some(outcome.error_reason());
                } else {
                    let serving2 = outcome
                        .envelope
                        .as_ref()
                        .and_then(|e| e.provider.as_deref());
                    let v2 = verify_pinned(serving2, endpoint);
                    if serving2.is_some() && !v2 {
                        invalid_reason = Some("pin_drift".to_string());
                    }
                }
            }
        }

        // 4. Tool-call validation: the model must call fast_tool with
        //    step == next_step.
        if invalid_reason.is_none() {
            match outcome.tool_call_step {
                Some(step) if step == next_step => {}
                Some(_) => invalid_reason = Some("tool_call_mismatch".to_string()),
                None => {
                    let finish = outcome
                        .envelope
                        .as_ref()
                        .and_then(|e| e.choices.first())
                        .and_then(|c| c.finish_reason.as_deref());
                    if finish == Some("length") {
                        let retry_body = build_request_body(
                            model,
                            base,
                            &frames,
                            &tag,
                            reasoning_effort,
                            LENGTH_RETRY_MAX_TOKENS,
                        );
                        outcome = send_round(client, key, &retry_body).await;
                        round_billed += usage_from(outcome.envelope.as_ref()).cost.unwrap_or(0.0);
                        if outcome.envelope.is_none() {
                            invalid_reason = Some(outcome.error_reason());
                        } else if outcome.tool_call_step != Some(next_step) {
                            invalid_reason = Some("no_tool_call".to_string());
                        }
                    } else {
                        invalid_reason = Some("no_tool_call".to_string());
                    }
                }
            }
        }

        let tool_call_id = outcome
            .tool_call_id
            .clone()
            .unwrap_or_else(|| format!("call_{r}"));
        let reasoning = outcome.reasoning.clone();

        let usage = usage_from(outcome.envelope.as_ref());
        // Response-cache-hit rounds are Invalid for bucketing and their cached tokens are
        // stale — they must not count toward the zero-cache predicate.
        if !outcome.response_cache_hit {
            cached_observations.push(usage.cached_tokens.unwrap_or(0));
        }
        if r == 0 {
            round0_prompt_tokens = usage.prompt_tokens;
        }
        let base_pt = round0_prompt_tokens.or(w2_prompt_tokens).unwrap_or(0);
        let expected =
            expected_cached_for_round(base_cached, usage.prompt_tokens.unwrap_or(0), base_pt);
        let classification = if invalid_reason.is_some() {
            CacheClassification::Invalid
        } else {
            classify_round(usage.cached_tokens.unwrap_or(0), expected)
        };
        let classification_str = classification.as_str();
        classifications.push(classification);
        nominal_gaps.push(nominal_gap_secs);

        let (over_guard, over_cap) = context.budget.record(&tag, round_billed);
        if over_guard {
            ladder_stop = Some(StopReason::SpendGuardTripped);
            break;
        }
        if over_cap {
            context.abort(AbortCause::SpendCap, context.budget.cap_exceeded_reason());
            ladder_stop = Some(StopReason::SpendCapTripped);
            break;
        }

        if invalid_reason.is_none() {
            frames.push(ToolFrame {
                id: tool_call_id,
                step: next_step,
                reasoning,
            });
        }

        eprintln!(
            "bench [{tag}] rung {r} gap={nominal_gap_secs}s {}",
            invalid_reason.as_deref().unwrap_or(classification_str),
        );

        // Deterministic tool delay; the deadline and the run-level abort both race the sleep.
        if r < ladder_secs.len() {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(ladder_secs[r])) => {}
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(context.deadline)) => {
                    ladder_stop = Some(StopReason::RunDeadlineReached);
                    break;
                }
                _ = context.aborted() => {
                    ladder_stop = Some(context.abort_stop());
                    break;
                }
            }
        }
    }

    let any_cached = cached_observations.iter().any(|&c| c > 0);
    // Derived, never tracked: every iteration that starts pushes exactly one classification, so
    // the ladder ran to its last rung iff the count matches. A guard or cap that trips on the
    // LAST rung therefore cannot mark a complete ladder truncated, and a future `break` needs no
    // bookkeeping for the flag. The flag and the bucket's suffix then come from the one predicate,
    // so a row can never be marked without saying so (or vice versa).
    let completed = classifications.len() == rounds;
    let truncated = is_truncated(completed, &classifications);
    let cache_hold_bucket =
        cache_hold_result(any_cached, &classifications, &nominal_gaps, completed);

    tracing::debug!(
        tag = %tag,
        elapsed_ms = run_started.elapsed().as_millis(),
        cache_hold_bucket = %cache_hold_bucket,
        truncated,
        "provider run finished"
    );

    // Settled last, so the guard still says "task died" if anything above panics.
    match stop_reason(completed, ladder_stop) {
        Some(reason) => stop_line.stop(reason),
        None => stop_line.complete(),
    }

    ProviderRun {
        tag,
        cache_hold_bucket,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> BasePrompt {
        BasePrompt {
            system: "sys".to_string(),
            user: "user text".to_string(),
        }
    }

    /// The request the harness sends: the chat shape (system + user, then one assistant/tool
    /// pair per frame, with the reasoning fields spread into the assistant message) and the
    /// pinned body — model, token cap, provider order, the single tool — with `reasoning_effort`
    /// only when one was chosen and `tool_choice` never sent.
    #[test]
    fn request_construction() {
        let frames = vec![
            ToolFrame {
                id: "call_0".to_string(),
                step: 0,
                reasoning: None,
            },
            ToolFrame {
                id: "call_1".to_string(),
                step: 1,
                reasoning: Some(json!({
                    "reasoning_content": "rc",
                    "reasoning": "r",
                    "reasoning_details": {"x": 1},
                })),
            },
        ];
        let msgs = build_messages(&base(), &frames);
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[1], json!({"role": "user", "content": "user text"}));
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "call_0");
        assert_eq!(msgs[2]["tool_calls"][0]["type"], "function");
        assert_eq!(
            msgs[2]["tool_calls"][0]["function"]["arguments"],
            "{\"step\":0}"
        );
        assert!(msgs[2].get("reasoning_content").is_none());
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "call_0");
        assert_eq!(msgs[4]["role"], "assistant");
        assert_eq!(msgs[4]["reasoning_content"], "rc");
        assert_eq!(msgs[4]["reasoning"], "r");
        assert_eq!(msgs[4]["reasoning_details"]["x"], 1);
        assert_eq!(
            msgs[4]["tool_calls"][0]["function"]["arguments"],
            "{\"step\":1}"
        );
        assert_eq!(msgs[5]["tool_call_id"], "call_1");

        let body = build_request_body("acme/model-1", &base(), &[], "acme/fp8", None, 128);
        assert_eq!(body["model"], "acme/model-1");
        assert_eq!(body["max_tokens"], 128);
        assert!(body.get("reasoning_effort").is_none());
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice must be omitted (providers default to auto)"
        );
        assert_eq!(
            body["provider"],
            json!({"order": ["acme/fp8"], "allow_fallbacks": false})
        );
        assert_eq!(body["tools"][0]["function"]["name"], "fast_tool");

        let with_effort = build_request_body("m", &base(), &[], "t", Some("low"), 256);
        assert_eq!(with_effort["reasoning_effort"], "low");
        assert!(with_effort.get("tool_choice").is_none());
    }

    #[test]
    fn raw_envelope_parse_fixture() {
        const FIXTURE: &str = r#"{
            "id": "gen-abc123",
            "created": 1755000000,
            "model": "acme/model-1",
            "provider": "Acme Cloud",
            "system_fingerprint": "fp-1",
            "service_tier": "default",
            "is_byok": false,
            "openrouter_metadata": {"elapsed": 42},
            "usage": {
                "prompt_tokens": 16000,
                "completion_tokens": 12,
                "total_tokens": 16012,
                "prompt_tokens_details": {"cached_tokens": 15800, "cache_write_tokens": 200},
                "completion_tokens_details": {"reasoning_tokens": 4},
                "prompt_cache_hit_tokens": 15800,
                "prompt_cache_miss_tokens": 200,
                "cost": 0.0001234,
                "cost_details": {"prompt": 0.0001, "completion": 0.0000234}
            },
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "fast_tool", "arguments": "{\"step\":0}"}
                    }],
                    "reasoning_content": "thinking",
                    "reasoning": "more",
                    "reasoning_details": {"tokens": 4}
                }
            }]
        }"#;
        let env: RawEnvelope = serde_json::from_str(FIXTURE).expect("fixture parses");
        assert_eq!(env.provider.as_deref(), Some("Acme Cloud"));
        let usage = env.usage.as_ref().expect("usage");
        assert_eq!(usage.cost, Some(0.000_123_4));
        let details = usage.prompt_tokens_details.as_ref().expect("details");
        assert_eq!(details.cached_tokens, Some(15800));
        let choice = &env.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        let call = choice.message.tool_calls.as_ref().expect("tool_calls")[0].clone();
        assert_eq!(
            call.function.as_ref().and_then(|f| f.name.as_deref()),
            Some("fast_tool")
        );
        assert_eq!(
            call.function.as_ref().and_then(|f| f.arguments.as_deref()),
            Some("{\"step\":0}")
        );
        let (step, id, reasoning) = extract_tool_call(Some(&env));
        assert_eq!(step, Some(0));
        assert_eq!(id.as_deref(), Some("call_0"));
        let reasoning = reasoning.expect("reasoning combined");
        assert_eq!(reasoning["reasoning_content"], "thinking");
        assert_eq!(reasoning["reasoning"], "more");
        assert_eq!(reasoning["reasoning_details"]["tokens"], 4);
    }

    #[test]
    fn retry_sleep_clamp_boundaries() {
        let s429 = reqwest::StatusCode::TOO_MANY_REQUESTS;
        assert_eq!(retry_sleep_ms(s429, None), 5000);
        assert_eq!(retry_sleep_ms(s429, Some(1000)), 5000);
        assert_eq!(retry_sleep_ms(s429, Some(30_000)), 30_000);
        assert_eq!(retry_sleep_ms(s429, Some(120_000)), 60_000);
        assert_eq!(
            retry_sleep_ms(reqwest::StatusCode::INTERNAL_SERVER_ERROR, None),
            5000
        );
    }

    #[test]
    fn classify_http_error_mapping() {
        assert_eq!(
            classify_http_error(reqwest::StatusCode::UNAUTHORIZED),
            Some("auth")
        );
        assert_eq!(
            classify_http_error(reqwest::StatusCode::PAYMENT_REQUIRED),
            Some("quota")
        );
        assert_eq!(
            classify_http_error(reqwest::StatusCode::TOO_MANY_REQUESTS),
            None
        );
        assert_eq!(
            classify_http_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            None
        );
        assert_eq!(
            classify_http_error(reqwest::StatusCode::BAD_REQUEST),
            Some("http_4xx")
        );
    }

    /// The budget arithmetic of the default ladder: eight gaps, the sum of their sleeps, eleven
    /// requests per endpoint (2 warmup + 9 ladder rounds, via the production count), and enough
    /// deadline above that plus the overhead margin for the run to finish instead of being killed.
    #[test]
    fn default_ladder_budget_fits_the_deadline() {
        assert_eq!(DEFAULT_LADDER_SECS.len(), 8, "eight gaps");
        let sleeps: u64 = DEFAULT_LADDER_SECS.iter().sum();
        assert_eq!(sleeps, 6455, "sum of the default ladder gaps");
        assert_eq!(
            crate::bench_openrouter::rounds_per_provider(DEFAULT_LADDER_SECS),
            11,
            "2 warmup + 9 ladder rounds"
        );

        // The deadline must leave real room above the sleeps: eleven requests' worth of
        // overhead (the slowest stored run spent ~306 s over ten) has to fit inside it.
        let margin = RUN_DEADLINE_MINS * 60 - sleeps;
        assert!(
            margin >= 900,
            "the deadline leaves {margin} s of overhead above the ladder sleeps"
        );
        const { assert!(RUN_DEADLINE_MINS >= 130) };
        const { assert!(OUTER_TIMEOUT_MINS >= RUN_DEADLINE_MINS + 30) };
    }

    /// The run's cause is the first one recorded: a provider that crosses the cap and then fails
    /// its warmup must not rename the abort. The cause class is recorded with the reason, so the
    /// stop line that names the class and the report that names the reason cannot disagree.
    #[test]
    fn the_first_abort_reason_wins() {
        let ctx = RunContext::new(RunBudget::new(1.0, 0.5), Instant::now());
        ctx.abort(AbortCause::SpendCap, "spend cap exceeded".to_string());
        ctx.abort(AbortCause::AuthRefused, "warmup failed (auth)".to_string());
        assert!(ctx.is_aborted(), "an abort cancels the token");
        assert_eq!(ctx.abort_reason().as_deref(), Some("spend cap exceeded"));
        assert_eq!(
            ctx.abort_stop(),
            StopReason::RunAborted(AbortCause::SpendCap)
        );
    }

    /// A warmup spend-guard or cap trip ends the ladder before its last rung with no observed
    /// drop, so the row is truncated — never a bare `not measured`, which would claim the endpoint
    /// produced nothing usable.
    #[test]
    fn a_warmup_guard_trip_is_a_truncated_row() {
        let run = ProviderRun::ladder_never_ran("acme".to_string());
        assert!(run.truncated);
        assert_eq!(run.cache_hold_bucket, "not measured (ladder truncated)");
        // The row that never produced a usable response is not truncated.
        let unmeasured = ProviderRun::not_measured("acme".to_string());
        assert!(!unmeasured.truncated);
        assert_eq!(unmeasured.cache_hold_bucket, "not measured");
    }

    /// The stop decision is "did not complete its ladder", never the truncation flag: a guard or
    /// cap trip on the FINAL rung breaks the loop with a reason but the ladder ran to its end, and
    /// a ladder cut by the deadline after an early drop is not truncated yet never finished.
    #[test]
    fn the_stop_decision_is_an_unfinished_ladder() {
        // A reason alongside a complete ladder — the final-rung guard or cap trip.
        assert_eq!(stop_reason(true, Some(StopReason::SpendGuardTripped)), None);
        assert_eq!(stop_reason(true, Some(StopReason::SpendCapTripped)), None);
        assert_eq!(stop_reason(true, None), None, "a finished ladder is quiet");
        // An unfinished ladder always gets its reason back, whatever it is.
        for (_, reason) in every_stop_reason() {
            assert_eq!(
                stop_reason(false, Some(reason.clone())),
                Some(reason.clone()),
                "an unfinished ladder reports {reason:?}"
            );
        }
        // The two must not be confused. An early drop settles the bucket, so a ladder afterwards
        // cut by the deadline is NOT truncated — and it still never finished, so it gets its line.
        let dropped_then_cut = [CacheClassification::Drop, CacheClassification::Invalid];
        assert!(
            !is_truncated(false, &dropped_then_cut)
                && stop_reason(false, Some(StopReason::RunDeadlineReached)).is_some(),
            "an early drop cut by the deadline is not truncated, yet never finished its ladder"
        );
        // And a full ladder whose final rung tripped the guard is complete and quiet.
        assert!(
            !is_truncated(true, &[CacheClassification::Hit; 3])
                && stop_reason(true, Some(StopReason::SpendGuardTripped)).is_none(),
            "a final-rung guard trip on a complete ladder gets no line"
        );
    }

    /// Every stop reason with the exact text its line must carry — the vocabulary the tests pin.
    fn every_stop_reason() -> Vec<(&'static str, StopReason)> {
        vec![
            (
                "warmup failed (transport)",
                StopReason::WarmupFailed("transport".to_string()),
            ),
            (
                "warmup failed (timeout)",
                StopReason::WarmupFailed("timeout".to_string()),
            ),
            (
                "warmup failed (auth)",
                StopReason::WarmupFailed("auth".to_string()),
            ),
            (
                "warmup failed (quota)",
                StopReason::WarmupFailed("quota".to_string()),
            ),
            (
                "warmup failed (http_5xx)",
                StopReason::WarmupFailed("http_5xx".to_string()),
            ),
            (
                "per-provider spend guard tripped",
                StopReason::SpendGuardTripped,
            ),
            ("total spend cap tripped", StopReason::SpendCapTripped),
            ("run deadline reached", StopReason::RunDeadlineReached),
            (
                "run aborted (total spend cap)",
                StopReason::RunAborted(AbortCause::SpendCap),
            ),
            (
                "run aborted (auth refusal)",
                StopReason::RunAborted(AbortCause::AuthRefused),
            ),
            (
                "run aborted (quota refusal)",
                StopReason::RunAborted(AbortCause::QuotaRefused),
            ),
            (
                "run aborted (outer timeout)",
                StopReason::RunAborted(AbortCause::OuterTimeout),
            ),
            ("task died without a result", StopReason::TaskDied),
        ]
    }

    /// Each stop reason's line: the per-endpoint prefix and stream of the rung lines, the class of
    /// the stop as its own text, and nothing measured. A warmup failure names its class, so a
    /// transport failure and a timeout (or a refusal and an HTTP error) never read alike.
    #[test]
    fn the_stop_line_names_the_tag_and_the_class() {
        let tag = "acme/fp8";
        for (text, reason) in every_stop_reason() {
            let line = reason.line(tag);
            assert_eq!(line, format!("bench [{tag}] stopped: {text}"));
            assert!(!line.contains(" rung "), "not a rung line: {line}");
        }
    }

    /// An aborted run's stop line names the cause class only: the abort's recorded reason holds a
    /// dollar amount and the aborting endpoint's tag, and neither may reach the line.
    #[test]
    fn an_abort_stop_line_drops_the_dollar_amount_and_the_foreign_tag() {
        let ctx = RunContext::new(RunBudget::new(1.0, 0.5), Instant::now());
        ctx.abort(
            AbortCause::SpendCap,
            "spend cap exceeded ($1.2345 > $1.0000)".to_string(),
        );
        let line = ctx.abort_stop().line("acme/fp8");
        assert_eq!(
            line,
            "bench [acme/fp8] stopped: run aborted (total spend cap)"
        );
        assert!(!line.contains('$'), "{line}");

        let ctx = RunContext::new(RunBudget::new(1.0, 0.5), Instant::now());
        ctx.abort(
            AbortCause::AuthRefused,
            "warmup failed (auth); provider 'other/endpoint'".to_string(),
        );
        let line = ctx.abort_stop().line("acme/fp8");
        assert_eq!(line, "bench [acme/fp8] stopped: run aborted (auth refusal)");
        assert!(!line.contains("other/endpoint"), "{line}");
        // The report still carries the full reason.
        assert_eq!(
            ctx.abort_reason().as_deref(),
            Some("warmup failed (auth); provider 'other/endpoint'")
        );
    }

    /// Only a refusal aborts the run; every other warmup class gates that endpoint's ladder alone.
    #[test]
    fn only_a_refusal_aborts_the_run() {
        assert_eq!(AbortCause::refusal("auth"), Some(AbortCause::AuthRefused));
        assert_eq!(AbortCause::refusal("quota"), Some(AbortCause::QuotaRefused));
        for class in [
            "transport",
            "timeout",
            "http_4xx",
            "http_5xx",
            "rate_limited",
            "parse",
        ] {
            assert_eq!(
                AbortCause::refusal(class),
                None,
                "{class} gates one endpoint"
            );
        }
    }

    /// The guard is armed pessimistically: a task that dies (a panic, a cancellation) writes its
    /// line, a settled stop writes the reason it recorded, and a completed ladder writes nothing.
    #[test]
    fn the_stop_guard_is_armed_as_a_death_until_the_ladder_settles() {
        let armed = StopLine::new("acme/fp8".to_string());
        assert_eq!(
            armed.pending().as_deref(),
            Some("bench [acme/fp8] stopped: task died without a result")
        );

        let mut stopped = StopLine::new("acme/fp8".to_string());
        stopped.stop(StopReason::RunDeadlineReached);
        assert_eq!(
            stopped.pending().as_deref(),
            Some("bench [acme/fp8] stopped: run deadline reached")
        );

        let mut completed = StopLine::new("acme/fp8".to_string());
        completed.complete();
        assert_eq!(completed.pending(), None, "a completed ladder is quiet");
    }
}
