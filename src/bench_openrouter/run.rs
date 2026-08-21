//! TTL-ladder executor for the OpenRouter provider benchmark.
//!
//! This is a STANDALONE bench HTTP path — it deliberately does NOT use
//! [`crate::providers::compatible`], [`crate::retry`], or the global CONFIG
//! provider stack. It owns its reqwest client (provided by the full-run
//! orchestrator), its own bounded non-jittered retry loop, and its own spend
//! accounting, so the measurement is independent of the production agent
//! pipeline.
//!
//! # Protocol
//!
//! - **Canonical deterministic requests**: every request body is built from
//!   the same base prompt (harness system + filler user text) plus the
//!   verified tool frames of prior rounds ([`build_request_body`] /
//!   [`build_messages`]). The prompt prefix is byte-identical across a run's
//!   warmup and ladder rounds, so provider-side prompt caches can be
//!   measured. `serde_json::json!` builds BTreeMap objects → sorted keys →
//!   canonical JSON ([`canonical_messages_json`]).
//! - **Warmup gate**: two byte-identical warmup requests (W1, W2) establish
//!   the base cache. W1 also gates auth/quota failures before any spend.
//! - **Cache-hold predicate**: a provider "does not cache" ONLY when its raw
//!   cached tokens are zero across the warmup-2 sample and every ladder round
//!   (warmup-1 is excluded — it is cross-run contaminated). Any non-zero
//!   cached observation means the provider caches, and its bucket comes from
//!   the ladder ([`crate::bench_openrouter::classify::cache_hold_result`]).
//!   Providers that failed to run (403, 429, etc.) are "not measured".
//! - **TTL ladder**: the ladder is a list of inactivity GAPS between ladder
//!   rounds (a 7-entry ladder = 8 rounds). Each round re-sends the grown
//!   conversation after its gap and compares the provider's reported cached
//!   tokens against the full-hold expectation
//!   ([`crate::bench_openrouter::classify::classify_round`]).
//! - **Bounded non-jittered retries**: at most 3 attempts per round; sleeps
//!   are fixed (Retry-After clamped into [5 s, 60 s], else 5 s) and NEVER
//!   jittered — jitter would corrupt the gap measurement.
//! - **Spend accounting**: per-round billed cost accumulates against a
//!   per-provider guard and a total cap ([`RunBudget`]); the guard stops this
//!   provider's ladder, the cap aborts the whole run.
//! - **Deadline + abort**: [`RUN_DEADLINE_MINS`] caps the whole run; the
//!   per-provider deadline and the run-level abort token (spend cap / auth /
//!   quota) both race every ladder sleep, so a stalled provider cannot hang
//!   the bench and an aborted run interrupts a sleeping provider promptly.
//! - **Pinning verification**: the serving provider from the response
//!   metadata is checked against the endpoint's names
//!   ([`crate::bench_openrouter::classify::verify_pinned`]); a drift is
//!   retried once and then marked Invalid (`"pin_drift"`).
//! - **Response-cache detection**: a `hit` in the `x-openrouter-cache-status`
//!   header means OpenRouter answered from its response cache without
//!   exercising the model — such a round is re-sent once after a short delay
//!   and marked Invalid (`"response_cache"`) if it persists.
//!
//! `run_provider` is driven by the full-run scheduler in [`crate::bench_openrouter`]
//! and returns a [`ProviderRun`] carrying the tag and the derived cache-hold
//! result.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;

use crate::bench_openrouter::classify::{
    CacheClassification, cache_hold_result, classify_round, expected_cached_for_round,
    verify_pinned,
};
use crate::bench_openrouter::discovery::EndpointInfo;
use crate::util::UnwrapPoison;

// ── Constants ──────────────────────────────────────────────────────

/// Whole-run deadline (minutes). The orchestrator derives the per-provider
/// deadline (a [`std::time::Instant`]) from the run start + this constant and
/// stores it in [`RunContext::deadline`]; the executor races it, it never
/// computes it.
pub(crate) const RUN_DEADLINE_MINS: u64 = 55;

/// OpenRouter chat-completions endpoint (the only HTTP call this module makes).
const CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Per-request total timeout (seconds).
const REQUEST_TIMEOUT_SECS: u64 = 120;

/// Max attempts per round (1 initial + up to 2 retries).
const MAX_ATTEMPTS: u32 = 3;

/// Retry sleep floor and ceiling (ms) — Retry-After is clamped into this
/// range; absent/other retryable statuses sleep the floor. Never jittered.
const RETRY_SLEEP_MIN_MS: u64 = 5_000;
const RETRY_SLEEP_MAX_MS: u64 = 60_000;

/// Delay before the response-cache re-send (seconds).
const RESPONSE_CACHE_RETRY_DELAY_SECS: u64 = 2;

/// `max_tokens` for warmup and normal ladder rounds (the model only emits one
/// small tool call).
const ROUND_MAX_TOKENS: u32 = 128;

/// `max_tokens` for the single bounded retry of a `finish_reason == "length"`
/// round (same prompt, larger budget).
const LENGTH_RETRY_MAX_TOKENS: u32 = 1024;

/// Fixed word list for the deterministic filler blob (~16 short words).
const FILLER_WORDS: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa",
];

// ── Prompt construction ────────────────────────────────────────────

/// The base prompt for one provider run: the harness system prompt plus the
/// user message. Byte-identical across all of the run's rounds so the prompt
/// prefix is cacheable.
#[derive(Clone)]
pub(crate) struct BasePrompt {
    pub system: String,
    pub user: String,
}

/// Build the base prompt from the harness system prompt, a per-run nonce and
/// the deterministic filler blob. The nonce is embedded in the user message so
/// different runs are distinguishable in the API logs while remaining
/// byte-identical within one run; the user message also carries the fixed
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

/// Deterministic filler blob: numbered lines from [`FILLER_WORDS`] joined with
/// '\n', stopping once the accumulated length reaches `prefix_chars` (the last
/// line may push slightly past it — a partial final line is fine). No
/// randomness, no wall clock — two calls with the same argument are
/// byte-identical.
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

/// One verified tool-call frame: the model called `fast_tool` with a step, and
/// the executor records the call id + any reasoning fields for the next
/// round's message history.
pub(crate) struct ToolFrame {
    pub id: String,
    pub step: u64,
    pub reasoning: Option<serde_json::Value>,
}

/// Build the message array for a round: `[system, user]` followed by one
/// assistant(tool_calls) + tool(result) pair per verified frame. When a frame
/// carries `reasoning`, its keys are spread into the assistant message
/// (`reasoning_content`, `reasoning`, `reasoning_details` — whatever the
/// provider reported). The step the round expects the model to call lives in
/// the last tool result ("proceed to step N+1") — it is NOT embedded in the
/// prompt prefix (that would change it per round and break the cache
/// measurement); the caller verifies the response against it separately.
#[must_use]
pub(crate) fn build_messages(base: &BasePrompt, frames: &[ToolFrame]) -> Vec<serde_json::Value> {
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

/// Build the canonical request body for one round. `provider.order` pins the
/// endpoint's tag slug with fallbacks off; `tool_choice` is intentionally
/// omitted (providers default to auto) so the request stays cacheable;
/// `reasoning_effort` is omitted entirely when `None` so byte-identical
/// requests across rounds stay byte-identical. The expected step is not in the
/// body either — it lives in the tool results, keeping the prompt prefix
/// byte-identical across rounds.
#[must_use]
pub(crate) fn build_request_body(
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

/// Canonical serialization of a message array — stable across calls (sorted
/// keys via BTreeMap, no whitespace) so byte-identity can be asserted.
// Kept for the byte-identity test (`canonical_messages_json_deterministic`);
// production code no longer hashes or logs the canonical form.
#[allow(dead_code)]
#[must_use]
pub(crate) fn canonical_messages_json(messages: &[serde_json::Value]) -> String {
    serde_json::to_string(messages).expect("serializing a message array cannot fail")
}

// ── Wire types (permissive Deserialize) ────────────────────────────

/// Prompt-token detail of a usage object.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// Raw `usage` object of a chat-completions response.
// Deliberately trimmed to what the run consumes (prompt, cached, cost); serde
// ignores the other wire fields (completion, totals, cache-miss, etc.).
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    #[serde(default)]
    pub cost: Option<f64>,
}

/// Raw chat-completions response envelope. Every field is optional / defaulted
/// — a missing or malformed field is a parse failure (recorded, not fatal).
// Deliberately trimmed: all remaining fields are consumed by `verify_pinned` /
// `usage_from` / the tool-call extraction; serde ignores the other wire fields
// (id, created, model, etc.).
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawEnvelope {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub usage: Option<RawUsage>,
    #[serde(default)]
    pub choices: Vec<RawChoice>,
}

/// One choice of a chat-completions response.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawChoice {
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub message: RawMessage,
}

/// The assistant message of a choice.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawMessage {
    #[serde(default)]
    pub tool_calls: Option<Vec<RawToolCall>>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_details: Option<serde_json::Value>,
}

/// One tool call of an assistant message.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<RawToolCallFunction>,
}

/// The `function` object of a tool call.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RawToolCallFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Result of one HTTP round (one [`send_round`] call, including its internal
/// retries).
// The outcome exposes only what the ladder consumes: the parsed envelope, the
// error class, the response-cache flag, and the extracted tool-call fields.
pub(crate) struct RoundOutcome {
    pub envelope: Option<RawEnvelope>,
    pub error_class: Option<String>,
    pub response_cache_hit: bool,
    pub tool_call_step: Option<u64>,
    pub tool_call_id: Option<String>,
    pub reasoning: Option<serde_json::Value>,
}

// ── HTTP round ─────────────────────────────────────────────────────

/// Sleep (ms) before retrying a failed round: 429 honors Retry-After clamped
/// into [5000, 60000] (5000 when absent); any other retryable status sleeps a
/// fixed 5000. Never jittered — jitter would corrupt the TTL gap measurement.
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

/// Map a non-2xx status to its error-class string; retryable statuses (429,
/// 5xx) return None so the caller retries instead of failing the round.
#[must_use]
fn classify_http_error(status: reqwest::StatusCode) -> Option<&'static str> {
    match status.as_u16() {
        401 => Some("auth"),
        402 => Some("quota"),
        // Retryable statuses — the caller retries instead of failing the round.
        429 | 500..=599 => None,
        _ => Some("http_4xx"),
    }
}

/// Send one round's request with bounded (never-jittered) retries and return
/// the outcome. Every failure is recorded inside the returned [`RoundOutcome`]
/// (never an `Err`): auth/quota/other-4xx set `error_class` without retrying,
/// retryable statuses (429/5xx) and transport/timeout errors are retried
/// internally, and a final failure still yields an outcome with the error
/// class set. The API key and request body are never logged; at most a
/// `tracing::debug!` per attempt.
///
/// Retry policy (3 attempts total):
/// - 429 → Retry-After clamped into [5 s, 60 s] (or 5 s), retry.
/// - 5xx → 5 s, retry.
/// - transport / timeout → `"transport"` / `"timeout"`, 5 s, retry.
/// - 401 → `"auth"`, 402 → `"quota"`, other 4xx → `"http_4xx"`, no retry.
/// - 2xx → `x-openrouter-cache-status` inspected for response-cache hits, body
///   parsed (parse failure → `"parse"`).
pub(crate) async fn send_round(
    client: &reqwest::Client,
    key: &str,
    body: &serde_json::Value,
) -> RoundOutcome {
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

    // Exhausted retryable statuses (429/5xx) leave error_class None — record a
    // concrete class so the caller can classify the round as failed.
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

/// Pull the first `fast_tool` tool call out of the envelope: its parsed
/// `step` argument, its call id, and a combined reasoning object (any of
/// `reasoning_content` / `reasoning` / `reasoning_details` present on the
/// first choice's message).
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

/// Combine the reasoning fields of a message into one object when any are
/// present (`reasoning_content`, `reasoning`, `reasoning_details`).
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

// ── Round usage ────────────────────────────────────────────────────

/// Aggregated usage of one round (derived from the raw envelope) — only the
/// fields the cache-hold predicate and the budget need.
#[derive(Debug, Clone, Default)]
pub(crate) struct RoundUsage {
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub cost: Option<f64>,
}

/// Cached tokens from a raw usage object: `prompt_tokens_details.cached_tokens`
/// with a fallback to the Anthropic-style `prompt_cache_hit_tokens`.
#[must_use]
fn raw_cached(u: &RawUsage) -> Option<u64> {
    u.prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .or(u.prompt_cache_hit_tokens)
}

/// Cached tokens reported by a round's envelope (0 when none/absent).
#[must_use]
fn cached_tokens_of(envelope: Option<&RawEnvelope>) -> u64 {
    envelope
        .and_then(|e| e.usage.as_ref())
        .and_then(raw_cached)
        .unwrap_or(0)
}

/// Derive [`RoundUsage`] from a parsed envelope (all-None when absent).
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

// ── Budget + run context ───────────────────────────────────────────

/// Spend accounting for the whole bench run: a per-provider guard and a total
/// cap, both in USD.
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

    /// Record `billed` USD against `tag`. Returns `(over_guard, over_cap)` —
    /// true when the provider's guard or the total cap is exceeded AFTER this
    /// record.
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
}

/// Shared context for one provider run: budget, abort token, abort reason and
/// the run deadline.
pub(crate) struct RunContext {
    pub budget: RunBudget,
    pub abort: tokio_util::sync::CancellationToken,
    pub abort_reason: std::sync::Mutex<Option<String>>,
    pub deadline: Instant,
}

// ── Per-provider run ───────────────────────────────────────────────

/// Result of one provider run: the tag and the derived cache-hold result.
pub(crate) struct ProviderRun {
    pub tag: String,
    pub cache_hold_bucket: String,
}

impl ProviderRun {
    /// A provider that never produced a measured cache-hold result (failed to
    /// run, skipped the ladder, or was aborted before the ladder).
    #[must_use]
    fn not_measured(tag: String) -> Self {
        Self {
            tag,
            cache_hold_bucket: "not measured".to_string(),
        }
    }
}

/// Run one provider's warmup + TTL ladder. Never panics on provider behavior:
/// every failure degrades to a [`ProviderRun::not_measured`] result (after the
/// auth/quota abort side-effects) or a ladder-derived cache-hold bucket.
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
) -> ProviderRun {
    let tag = endpoint.tag.clone();

    // ── W1: warmup 1 — the auth/quota gate before any spend ──
    let w_body = build_request_body(model, base, &[], &tag, reasoning_effort, ROUND_MAX_TOKENS);
    let outcome_w1 = send_round(client, key, &w_body).await;

    // A failed W1 round gates the ladder: auth/quota abort the whole run; any
    // other error class skips this provider's ladder without touching the run
    // token (no wasted W2 + ladder spend).
    if let Some(class) = outcome_w1.error_class.as_deref() {
        if class == "auth" || class == "quota" {
            let reason = format!("warmup failed ({class}); provider '{tag}'");
            context.abort.cancel();
            *context.abort_reason.lock().unwrap_poison() = Some(reason.clone());
            tracing::debug!(tag = %tag, class, "provider aborted during warmup");
            return ProviderRun::not_measured(tag);
        }
        tracing::debug!(tag = %tag, class, "provider failed warmup; skipping ladder");
        return ProviderRun::not_measured(tag);
    }

    // ── W2: warmup 2 — byte-identical; the base-cache sample ──
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

    // Warmup spend counts against the budget: the cap covers ALL billed usage
    // (billed usage.cost accumulated against the cap), not just
    // ladder rounds. A provider whose warmup alone exceeds its guard is
    // pathological — record it and skip the ladder.
    let w_cost = usage_from(outcome_w1.envelope.as_ref()).cost.unwrap_or(0.0)
        + usage_from(outcome_w2.envelope.as_ref()).cost.unwrap_or(0.0);
    let (over_guard, over_cap) = context.budget.record(&tag, w_cost);
    if over_cap {
        let total = context.budget.total();
        let reason = format!(
            "spend cap exceeded (${total:.4} > ${:.4})",
            context.budget.cap_usd
        );
        context.abort.cancel();
        *context.abort_reason.lock().unwrap_poison() = Some(reason.clone());
        return ProviderRun::not_measured(tag);
    }
    if over_guard {
        return ProviderRun::not_measured(tag);
    }

    // W2 failure gates the ladder symmetrically with W1: auth/quota abort
    // the whole run; any other error class skips this provider's ladder
    // (running it with base_cached=0 would mis-scale every classification and
    // burn the full ladder on likely-failing rounds).
    if let Some(class) = outcome_w2.error_class.as_deref() {
        if class == "auth" || class == "quota" {
            let reason = format!("warmup failed ({class}); provider '{tag}'");
            context.abort.cancel();
            *context.abort_reason.lock().unwrap_poison() = Some(reason.clone());
            tracing::debug!(tag = %tag, class, "provider aborted during warmup 2");
            return ProviderRun::not_measured(tag);
        }
        tracing::debug!(tag = %tag, class, "provider failed warmup 2; skipping ladder");
        return ProviderRun::not_measured(tag);
    }

    // ── Ladder ──
    let rounds = ladder_secs.len() + 1;
    let mut frames: Vec<ToolFrame> = Vec::new();
    let mut classifications: Vec<CacheClassification> = Vec::new();
    let mut nominal_gaps: Vec<f64> = Vec::new();
    let mut round0_prompt_tokens: Option<u64> = None;

    for r in 0..rounds {
        if context.abort.is_cancelled() {
            break;
        }
        if Instant::now() >= context.deadline {
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
            Some(0.0)
        } else {
            Some(ladder_secs[r - 1] as f64)
        };

        let mut outcome = send_round(client, key, &body).await;
        let mut round_billed = usage_from(outcome.envelope.as_ref()).cost.unwrap_or(0.0);
        let mut invalid_reason: Option<String> = None;

        // 1. Envelope present? (send_round already recorded the error class.)
        if outcome.envelope.is_none() {
            invalid_reason = Some(
                outcome
                    .error_class
                    .clone()
                    .unwrap_or_else(|| "http_error".to_string()),
            );
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
                invalid_reason = Some(
                    outcome
                        .error_class
                        .clone()
                        .unwrap_or_else(|| "http_error".to_string()),
                );
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
                    invalid_reason = Some(
                        outcome
                            .error_class
                            .clone()
                            .unwrap_or_else(|| "http_error".to_string()),
                    );
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
                        // ONE bounded retry with a larger budget (same prompt).
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
                            invalid_reason = Some(
                                outcome
                                    .error_class
                                    .clone()
                                    .unwrap_or_else(|| "http_error".to_string()),
                            );
                        } else if outcome.tool_call_step != Some(next_step) {
                            invalid_reason = Some("no_tool_call".to_string());
                        }
                    } else {
                        invalid_reason = Some("no_tool_call".to_string());
                    }
                }
            }
        }

        // Tool-call frame for the next round (kept only when verified).
        let tool_call_id = outcome
            .tool_call_id
            .clone()
            .unwrap_or_else(|| format!("call_{r}"));
        let reasoning = outcome.reasoning.clone();

        // 5. Cache classification (ladder rounds only).
        let usage = usage_from(outcome.envelope.as_ref());
        // Response-cache-hit rounds are Invalid for bucketing and their cached
        // tokens are stale — they must not count toward the zero-cache predicate.
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
        nominal_gaps.push(nominal_gap_secs.unwrap_or(0.0));

        // 6. Spend accounting.
        let (over_guard, over_cap) = context.budget.record(&tag, round_billed);
        if over_guard {
            break;
        }
        if over_cap {
            let total = context.budget.total();
            let reason = format!(
                "spend cap exceeded (${total:.4} > ${:.4})",
                context.budget.cap_usd
            );
            context.abort.cancel();
            *context.abort_reason.lock().unwrap_poison() = Some(reason.clone());
            break;
        }

        // Verified frame for the next round (kept only when the round was valid).
        if invalid_reason.is_none() {
            frames.push(ToolFrame {
                id: tool_call_id,
                step: next_step,
                reasoning,
            });
        }

        eprintln!(
            "bench [{tag}] rung {r} gap={}s {}",
            nominal_gap_secs.unwrap_or(0.0),
            invalid_reason.as_deref().unwrap_or(classification_str),
        );

        // Deterministic tool delay; the deadline and the run-level abort both
        // race the sleep.
        if r < ladder_secs.len() {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(ladder_secs[r])) => {}
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(context.deadline)) => {
                    break;
                }
                _ = context.abort.cancelled() => {
                    break;
                }
            }
        }
    }

    // ── Result ──
    let any_cached = cached_observations.iter().any(|&c| c > 0);
    let cache_hold_bucket = cache_hold_result(any_cached, &classifications, &nominal_gaps);

    tracing::debug!(
        tag = %tag,
        elapsed_ms = run_started.elapsed().as_millis(),
        cache_hold_bucket = %cache_hold_bucket,
        "provider run finished"
    );

    ProviderRun {
        tag,
        cache_hold_bucket,
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

    #[test]
    fn build_messages_shape_and_reasoning_spread() {
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
        // system + user + 2 × (assistant + tool)
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[1], json!({"role": "user", "content": "user text"}));
        // Frame 0: plain assistant + tool result.
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
        assert_eq!(msgs[3]["content"], "step 0 acknowledged; proceed to step 1");
        // Frame 1: reasoning spread into the assistant message.
        assert_eq!(msgs[4]["role"], "assistant");
        assert_eq!(msgs[4]["reasoning_content"], "rc");
        assert_eq!(msgs[4]["reasoning"], "r");
        assert_eq!(msgs[4]["reasoning_details"]["x"], 1);
        assert_eq!(
            msgs[4]["tool_calls"][0]["function"]["arguments"],
            "{\"step\":1}"
        );
        assert_eq!(msgs[5]["content"], "step 1 acknowledged; proceed to step 2");
    }

    #[test]
    fn build_request_body_pins_and_omits_effort_and_tool_choice() {
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
    fn generate_filler_deterministic_and_sized() {
        let a = generate_filler(4000);
        let b = generate_filler(4000);
        assert_eq!(a, b);
        assert!(a.len() >= 4000, "filler too short: {}", a.len());
        assert!(a.len() <= 4400, "filler overshoot: {}", a.len());
        assert_eq!(a.lines().next(), Some("000000 alpha"));
        assert!(a.contains('\n'));
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
        // Tool-call extraction: step, id, combined reasoning object.
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

    #[test]
    fn canonical_messages_json_deterministic() {
        let msgs = build_messages(&base(), &[]);
        let a = canonical_messages_json(&msgs);
        let b = canonical_messages_json(&msgs);
        assert_eq!(a, b);
        assert!(a.contains("\"content\":\"sys\""));
        assert!(a.contains("\"role\":\"system\""));
        // Round 0's body must be byte-identical to W2's (the base prompt only).
        let body = build_request_body("m", &base(), &[], "t", None, 128);
        assert_eq!(
            canonical_messages_json(body["messages"].as_array().expect("messages")),
            a
        );
    }
}
