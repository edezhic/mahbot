//! Provider-backed video transcription for channel enrichment: a video (via the ephemeral upload
//! bridge's public URL) becomes a text description, so the main agent loop only ever sees text. Not
//! the document converter and not local speech recognition — microphone audio on macOS goes through
//! `crate::audio`.

use crate::util::UnwrapPoison;

/// The transcriber for one configured endpoint and model. Its API key comes from the live config
/// at request time ([`bearer_auth_header()`](crate::util::http::bearer_auth_header)), so config
/// reloads take effect without recreating the transcriber.
#[derive(Clone)]
pub(crate) struct MediaTranscriber {
    api_url: String,
    model: String,
}

/// Overall cap for one video transcription (upload + model call): bounds the fail-open stall of
/// both callers well under the worst case of the upload bridge's per-host timeouts.
const VIDEO_TRANSCRIPTION_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(3);

impl MediaTranscriber {
    #[must_use]
    pub(crate) fn new(api_url: String, model: String) -> Self {
        Self { api_url, model }
    }

    fn chat_url(&self) -> String {
        crate::providers::ensure_chat_completions_url(&self.api_url)
    }

    /// Perform the model call and parse the assistant text, WITHOUT any live or durable telemetry
    /// — the caller owns recording. When `marker` is given it is stamped immediately before the
    /// request, so the caller's cap-drop path can attribute a timeout to the LLM call.
    async fn transcribe_media_raw(
        &self,
        content_part: serde_json::Value,
        marker: Option<&ModelCallMarker>,
    ) -> anyhow::Result<RawTranscription> {
        let prompt = crate::prompt::load_prompt("media_transcription.md");
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        content_part
                    ]
                }
            ],
            "max_tokens": 2048,
            "reasoning": {"enabled": false},
        });

        if let Some(marker) = marker {
            marker.mark();
        }

        // NOTE: `post_json_to_provider` returns non-2xx responses as typed
        // [`HttpError`](crate::util::error::HttpError) (via `downcast_ref`). Safe because the
        // fail-open callers log a warning and fall back — it never reaches the retry logic in the
        // provider layer.
        let result =
            crate::util::http::post_json_to_provider(&self.chat_url(), &body, "transcription")
                .await?;

        let finish_reason = result["choices"][0]["finish_reason"]
            .as_str()
            .map(str::to_string);
        let text = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        if text.is_empty() {
            return Ok(RawTranscription::EmptyContent);
        }

        // Transcription text is embedded in tool outcomes and enriched messages that the trusted
        // marker scans parse — neutralize any marker-shaped substring the model quoted from the
        // media (the detailed prompt captures on-screen text) so it cannot be misparsed.
        Ok(RawTranscription::Success {
            text: scrub_marker_like(&text),
            finish_reason,
        })
    }

    /// Resolve the live-view tracker and the durable-call carrier for a transcription call: the
    /// agent task-local wins when present (in-agent path), else the explicit `workspace` (inbound
    /// path). Exactly one live entry is ever created per call.
    fn transcription_context(
        &self,
        workspace: Option<&str>,
    ) -> (Option<LiveTrackingGuard>, crate::stats::LlmCallMeta) {
        let tracking = crate::agent::CURRENT_TOOL_AGENT_TRACKING
            .try_with(Clone::clone)
            .ok()
            .flatten();
        let live = match &tracking {
            Some(t) => Some(LiveTrackingGuard::Agent {
                _guard: crate::agent::registry::AGENT_REGISTRY.activity_started(
                    &t.agent_id,
                    t.generation,
                    "transcribing",
                ),
            }),
            None => workspace.map(|ws| LiveTrackingGuard::Call {
                _guard: crate::agent::registry::NON_AGENT_CALLS.register(
                    "media_transcription",
                    ws,
                    None,
                    false,
                    None,
                ),
            }),
        };
        let call = crate::stats::LlmCallMeta {
            meta: crate::ChatRequestMeta {
                purpose: "media_transcription",
                agent_id: tracking
                    .as_ref()
                    .map(|t| t.agent_id.clone())
                    .unwrap_or_default(),
                role: tracking
                    .as_ref()
                    .map(|t| t.role.clone())
                    .unwrap_or_default(),
                workspace: tracking.as_ref().map_or_else(
                    || workspace.unwrap_or_default().to_string(),
                    |t| t.workspace.clone(),
                ),
                ticket_id: None,
            },
            model: self.model.clone(),
            provider_order: None,
        };
        (live, call)
    }
}

/// Raw outcome of one media-transcription model call — telemetry-free, so the caller owns the
/// durable record (a video cap-drop can then never lose or duplicate the row).
enum RawTranscription {
    /// Model returned non-empty assistant text.
    Success {
        text: String,
        finish_reason: Option<String>,
    },
    /// Model returned empty content (treated as a failure).
    EmptyContent,
}

/// Shared marker between the capped video-transcription future and the cap-drop path: stamped
/// when the model call launches, so the drop path attributes a timeout to the LLM call rather
/// than the upload phase, and measures the model-call duration honestly.
struct ModelCallMarker(std::sync::Mutex<Option<std::time::Instant>>);

impl ModelCallMarker {
    fn new() -> Self {
        Self(std::sync::Mutex::new(None))
    }

    /// Stamp the model-call launch; called immediately before the HTTP request is sent.
    fn mark(&self) {
        *self.0.lock().unwrap_poison() = Some(std::time::Instant::now());
    }

    fn started_at(&self) -> Option<std::time::Instant> {
        self.0.lock().unwrap_poison().as_ref().copied()
    }
}

/// Record the rows for one media-transcription call (fail-open; usage columns stay NULL,
/// retry_attempts always 1). A failure also persists one `llm_failures` row under a fresh
/// operation id.
#[expect(clippy::cast_possible_truncation)]
fn record_transcription(
    call: &crate::stats::LlmCallMeta,
    started: std::time::Instant,
    finish_reason: Option<&str>,
    failure_class: Option<&'static str>,
    error_chain: Option<&str>,
) {
    crate::stats::record_llm_operation_meta(
        call,
        started.elapsed().as_millis() as u64,
        1,
        None,
        finish_reason,
        failure_class,
    );
    if let (Some(class), Some(chain)) = (failure_class, error_chain) {
        let operation_id = crate::generate_id();
        crate::stats::record_llm_attempt_failure_meta(call, &operation_id, class, chain, started);
    }
}

/// Record the durable rows for a raw transcription outcome and resolve it to text (or the
/// original error). The single place that maps outcomes to `llm_requests`/`llm_failures`, so
/// every call records exactly once.
fn finish_transcription(
    call: &crate::stats::LlmCallMeta,
    started: std::time::Instant,
    outcome: Result<RawTranscription, anyhow::Error>,
) -> anyhow::Result<String> {
    match outcome {
        Ok(RawTranscription::Success {
            text,
            finish_reason,
        }) => {
            record_transcription(call, started, finish_reason.as_deref(), None, None);
            Ok(text)
        }
        Ok(RawTranscription::EmptyContent) => {
            record_transcription(
                call,
                started,
                None,
                Some(crate::retry::FailureClass::NoResponse.label()),
                Some("media transcription returned empty content"),
            );
            anyhow::bail!("media transcription returned empty content");
        }
        Err(e) => {
            // Classify through the canonical error cascade so 4xx client errors (auth, quota) land
            // as `non_retryable` instead of the blanket `transport` label.
            let failure = crate::providers::failure_class(
                crate::providers::error_class::classify_err(&e),
                false,
            );
            let chain = format!("{e:#}");
            record_transcription(call, started, None, Some(failure.label()), Some(&chain));
            Err(e)
        }
    }
}

/// One live-view tracker per media-transcription call — either the owning agent's activity
/// indicator (in-agent path) or a non-agent call row (inbound path). Exactly one variant is live
/// at a time; the guard clears it on every exit path.
///
/// `_guard` is held purely for its RAII Drop side effect and never read.
enum LiveTrackingGuard {
    Agent {
        _guard: crate::agent::registry::ActivityGuard,
    },
    Call {
        _guard: crate::agent::registry::NonAgentCallGuard,
    },
}

/// Transcribe a local video file — the shared fail-open helper for the inbound-enrichment
/// annotation and the video tool-result paths. Uploads the file via the ephemeral host bridge and
/// describes the returned public URL, returning `None` on unavailable transcriber, unsupported
/// format, upload/model failure or empty output, so callers degrade to their plain annotation.
/// The whole flow is capped by [`VIDEO_TRANSCRIPTION_TIMEOUT`].
///
/// `workspace` names the workspace for telemetry and for the live non-agent call row when no
/// agent context is present.
///
/// The live tracker and the durable record are owned by THIS scope — the capped future only
/// performs the model call, telemetry-free ([`MediaTranscriber::transcribe_media_raw`]). A cap
/// drop therefore cannot race the inner future's write: one launched model call produces exactly
/// one `llm_requests` row, and upload-phase failures record nothing.
pub(crate) async fn transcribe_video_file(
    path: &std::path::Path,
    workspace: Option<&str>,
) -> Option<String> {
    let transcriber = crate::providers::media_transcriber()?;
    if !crate::util::is_transcribable_video(path) {
        tracing::debug!(
            path = %path.display(),
            "Video format not supported by the transcription provider — skipping"
        );
        return None;
    }
    let (_live, call) = transcriber.transcription_context(workspace);
    let marker = ModelCallMarker::new();
    let outcome = tokio::time::timeout(VIDEO_TRANSCRIPTION_TIMEOUT, async {
        let url = crate::util::upload_bridge::upload_video_ephemeral_typed(path).await?;
        let content_part = serde_json::json!({"type": "video_url", "video_url": {"url": url}});
        transcriber
            .transcribe_media_raw(content_part, Some(&marker))
            .await
    })
    .await;
    // Duration measured from the model call's launch (consistent with every other transcription
    // row); the upload phase is not an LLM call.
    let started = marker.started_at().unwrap_or_else(std::time::Instant::now);
    if let Ok(inner) = outcome {
        // An upload-phase failure never reaches the model call (the marker is unstamped) — record
        // nothing for it: no LLM request was made, so a durable row would be a phantom.
        if marker.started_at().is_none() {
            if let Err(e) = inner {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "Video transcription failed before the model call"
                );
            }
            return None;
        }
        match finish_transcription(&call, started, inner) {
            Ok(text) => Some(text),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "Video transcription failed");
                None
            }
        }
    } else {
        // The cap fired. If the model call had launched, the dropped attempt is a failed LLM call
        // and is recorded durably here (the inner future never records, so this cannot duplicate a
        // row). A timeout during the upload phase records nothing.
        if marker.started_at().is_some() {
            record_transcription(
                &call,
                started,
                None,
                Some(crate::retry::FailureClass::Transport.label()),
                Some("video transcription timed out"),
            );
        }
        tracing::warn!(path = %path.display(), "Video transcription timed out");
        None
    }
}

/// Neutralize marker-shaped substrings (e.g. `[IMAGE:`, `[video:`) in LLM transcription text so
/// the media-marker scans cannot treat quoted text as a real reference. Case-insensitive: the
/// Telegram mirror path's marker regex matches any case.
fn scrub_marker_like(text: &str) -> String {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\[(image|audio|video):").expect("marker scrub regex must compile")
    });
    RE.replace_all(text, "($1:").to_string()
}
