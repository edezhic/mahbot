# Retry policy for scoped LLM operations (mahbot-1066)

## Scope

Two expensive code paths use a hardened **outer retry loop** (a single retry
authority, configurable, hot-reloadable):

1. **Verdict extraction** — structured pass/fail verdicts from Analyst /
   Reviewer / QA agents (`run_parallel_agents` → `Agent::extract_verdict` →
   `extraction::retry_extract_structured_scoped`) and the Sanitation verdict
   (`dispatch_sanitation`).
2. **Analyst consolidation** — the synthesis of the 3 parallel analyst reports
   in the ask tool (`consolidate_analyst_responses` → `retry::retry_chat`).

Everything else (including workspace diagnostics discovery at
`workspace.rs:267`, which keeps its 3-attempt legacy loop) is unchanged.

## Why this exists

Server-side premature termination / truncation of non-streaming chat-completion
response bodies at the OpenRouter gateway / pinned DeepSeek upstream. The two
observed error strings are the SAME defect class:

- `"OpenRouter error reading response body"` — a cut violating HTTP framing.
- `"chat completions parse error: EOF while parsing a value at line N"` — a
  cut landing on a framing boundary.

Recovery is burst-shaped: same-parameter provider retries fail 15/15 within a
bad window, but minute-scale-spaced outer retries recover the class.

## Failure modes on scoped calls

| Class | Meaning | Retried? |
|-------|---------|----------|
| `transport` | Network/transport error, 5xx, 429, pre-header (TTFB) idle timeout | yes, byte-identical |
| `truncated_envelope` | Body-read error, content-length mismatch, or truncated error body | yes, byte-identical |
| `parse` | Full body, invalid JSON (LLM format issue) | yes, via re-prompt |
| `out_of_range_score` | Parsed but score ∉ [0,10] | yes, via re-prompt |
| `no_response` | Empty response text | consolidation: yes, byte-identical; extraction: funnels empty text into the parse/re-prompt branch |
| `non_retryable` | Auth, quota, invalid model, tool schema | **no — abort** |
| `shutdown` / `wall_clock_exceeded` | Global shutdown / budget exhausted | no — abort |

## Schedule (defaults)

- **Max attempts**: 7 (`retry_max_attempts`)
- **Backoff**: 5 / 10 / 20 / 40 / 60 / 90 s (canonical literal;
  `retry_base_backoff_ms` = 5000, `retry_max_backoff_ms` = 90000)
- **Custom base/cap**: changing EITHER `retry_base_backoff_ms` or
  `retry_max_backoff_ms` opts out of the pinned literal and into pure doubling
  capped at the cap — e.g. base 10000 / cap 90000 → 10/20/40/80/90; base 5000 /
  cap 60000 → 5/10/20/40/60; base 5000 / cap 120000 → 5/10/20/40/80/120. The
  deliberate 60 s tail exists only in the default configuration.
- **Jitter**: ±25% on the sleep ONLY — never on request bytes
- **Retry-After**: honored, clamped [5000 ms, 90000 ms], no jitter — applies
  only to the sleep following the 429 that carried it (stale values are
  cleared on subsequent non-429 failures)
- **Shutdown-abortable** via the global shutdown token
- **Wall-clock cap**: 600 s (`operation_timeout_secs`), authoritative over
  attempt count — the 600 s deadline binds before attempt 7 whenever failures
  are slow; that is correct behavior. The cap cannot be overshot by a backoff
  interval (every inter-attempt sleep is capped at the remaining budget) or by
  a stalled body read (every chunk wait is bounded by the idle timeout AND the
  remaining budget, whichever is tighter).

## Byte-identical retry parameters

All request parameters (model, messages, tools, max_tokens 32K, temperature,
reasoning_effort, provider routing) are byte-identical across ALL attempts:

- **temperature** is a compile-time constant (not hot-reloadable, no config).
- **reasoning_effort** is FIXED (never lowered — prefix-cache preservation is
  a user mandate; the cache is vital for cost).
- The ONLY permitted request mutation is the parse-failure re-prompt in the
  extraction path (appended messages, extends the cached prefix).
- Provider-internal retries are suppressed on scoped calls
  (`Provider::chat_scoped` bypasses `ReliableProvider`'s retry loop), so total
  provider HTTP calls per operation are explicitly bounded:
  **7 per verdict extraction, 7 per consolidation, 21 per 3-analyst wave**.
- Config tunables are snapshotted at operation start — hot-reload mid-retry
  does not change the schedule.

## Verdict integrity

Score ∈ [0,10] is enforced fail-closed: an out-of-range score is a parse
failure (retryable via re-prompt); if all attempts yield out-of-range scores
the operation FAILS with an explicit `out_of_range_score` classification — a
garbage score never passes any gate. Verdict thresholds (7/10 analysts,
9/10 verifiers) and phase-transition semantics are unchanged.

## Failure surfaces

### Verdict extraction (Analyst/Reviewer/QA/Sanitation) — fail-closed

Gates do not pass. Amendment B: on final failure, the ticket comment carries
the raw last-attempt agent response (diagnosability only) under the
`analyst_{i}` / `reviewer_{i}` / `qa_{i}` / sanitation role, sandwich-truncated
at ~24,000 bytes with an explicit `(N bytes omitted)` marker. Edge cases:

- No response → existing no-response template.
- Tool-call final attempt → "final attempt was a tool call — no text response".
- Transport-error final failure (no in-loop text) → per-attempt trail +
  classification in the comment.

### Analyst consolidation (ask tool) — fail-open (Amendment A)

On ANY consolidation failure (retry exhaustion OR immediate non-retryable
error), the raw VALID analyst reports are delivered with an explicit
`(unconsolidated — consolidation failed: {reason})` marker instead of an
error — findings are never lost. Cases: 0 valid → unchanged error; 1 valid →
unchanged raw passthrough; 2–3 valid + success → unchanged synthesized output;
2–3 valid + failure → raw dump. The result is free-form text consumed only by
LLMs; nothing parses it. The sync path's 5,000-byte tool-output truncation
applies (still better than an error).

## Telemetry

Every failed attempt is persisted to the dedicated `retry_failures` table in
`stats.db` (NOT `sessions.messages`; `logs.db` is corrupt since 2026-07-31 and
is not a reliable sink). Columns: attempt, failure class, full error cause
chain, HTTP version, content-length vs actual body length, content-encoding,
transfer-encoding, elapsed ms, first + last 200 bytes of the body, finish_reason,
usage.completion_tokens, retry_after_ms, recorded_at.

Content-length vs actual comparison: **equal ⇒ the server finalized a short
body; shorter ⇒ framing anomaly** (truncated envelope).

## Timeout semantics

Scoped calls use a dedicated HTTP client with NO total request timeout. The
send phase (request write + wait for response headers / TTFB) is bounded by the
60 s idle timeout (or the remaining operation budget, whichever is tighter), so
a pre-header server stall cannot consume the whole 600 s budget on attempt 1 —
the burst-shaped retry recovery still gets its later attempts. An idle (read)
timeout of 60 s resets while data flows (chunk-by-chunk body reads); every
chunk wait is bounded by the idle timeout AND the remaining operation budget,
whichever is tighter, so a stalled chunk classifies as `wall_clock_exceeded`
(rather than `truncated_envelope`) when the wall budget ran out mid-read and
cannot overshoot the operation cap. Non-scoped calls keep the existing 120 s
total request timeout, unchanged. Inter-attempt sleeps are additionally capped
at the remaining operation budget, so the wall-clock cap is never overshot by a
backoff interval.

## Configuration

Flat `config_kv` string fields, hot-reloadable, one per tunable — SHARED
across both scoped paths:

| Key | Default | Meaning |
|-----|---------|---------|
| `retry_max_attempts` | `7` | Loop attempts (provider HTTP calls per operation) |
| `retry_base_backoff_ms` | `5000` | Base backoff in ms |
| `retry_max_backoff_ms` | `90000` | Backoff cap in ms |
| `operation_timeout_secs` | `600` | Whole-operation wall-clock cap in seconds |

Defaults land behavior with zero config. Invalid values fall back to defaults.
Editable in the GUI Settings → "Retry" section.

## Out of scope (companion)

- Body healing (envelope-level repair, chunked reads, repeat-truncation
  detector): separate small ticket, lands after this one (its detector needs
  this ticket's telemetry).
- Provider-fallback widening / provider change: config-only operator decisions
  (user mandate: stay on OpenRouter; direct DeepSeek API off the table).
- Analyst `llm_loop` hardening: follow-up if truncation persists.
- Diagnostics extraction behavior: unchanged.
- Credential scrubbing / comment markdown-escaping: pre-existing conditions.
