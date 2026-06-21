//! SSE streaming parser for OpenAI-compatible providers.
//!
//! Extracted from `compatible.rs` — shared by all OpenAI-compatible providers.

use crate::{
    Reasoning, StreamChunk, StreamError, StreamEvent, StreamResult, ToolCall as ProviderToolCall,
    providers::compatible::ApiToolCallFunction, providers::compatible::parse_tool_call_arguments,
};
use futures_util::StreamExt;
use serde::Deserialize;

/// Server-Sent Event stream chunk for OpenAI-compatible streaming.
#[derive(Debug, Deserialize)]
struct StreamChunkResponse {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking models may stream output via `reasoning_content`.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_details: Option<serde_json::Value>,
    /// Native tool-calling deltas in `OpenAI` chat-completions streaming format.
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ApiToolCallFunction>,
    // Compatibility: some providers stream name/arguments at top-level.
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct StreamToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl StreamToolCallAccumulator {
    fn apply_delta(&mut self, delta: &StreamToolCallDelta) {
        if let Some(id) = delta.id.as_ref().filter(|value| !value.is_empty()) {
            self.id = Some(id.clone());
        }

        let delta_name = delta
            .function
            .as_ref()
            .and_then(|function| function.name.as_ref())
            .or(delta.name.as_ref())
            .filter(|value| !value.is_empty());
        if let Some(name) = delta_name {
            self.name = Some(name.clone());
        }

        if let Some(arguments_delta) = delta
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_ref())
            .or(delta.arguments.as_ref())
            .filter(|value| !value.is_empty())
        {
            self.arguments.push_str(arguments_delta);
        }
    }

    fn into_provider_tool_call(self) -> Option<ProviderToolCall> {
        let name = self.name?;
        let arguments = if self.arguments.trim().is_empty() {
            "{}".to_string()
        } else {
            self.arguments
        };
        let parsed = parse_tool_call_arguments(&name, &arguments);

        Some(ProviderToolCall {
            id: self.id.unwrap_or_else(crate::generate_id),
            name,
            arguments: parsed,
        })
    }
}

fn parse_sse_chunk(line: &str) -> StreamResult<Option<StreamChunkResponse>> {
    let line = line.trim();

    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }

    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data = data.trim();

    if data == "[DONE]" {
        return Ok(None);
    }

    serde_json::from_str(data)
        .map(Some)
        .map_err(StreamError::Json)
}

fn reasoning_stream_chunk(choice: &StreamChoice) -> Option<StreamChunk> {
    let reasoning = choice.delta.reasoning.clone();
    let reasoning_content = choice.delta.reasoning_content.clone();
    if reasoning.is_none() && reasoning_content.is_none() {
        return None;
    }
    Some(StreamChunk {
        delta: String::new(),
        reasoning: Some(Reasoning {
            reasoning,
            reasoning_content,
            reasoning_details: None,
        }),
    })
}

/// OpenAI-compatible streaming finishes native tool calling with `tool_calls` (or legacy
/// `function_call`). Do **not** emit on the first delta chunk — `function.name` often arrives
/// before `function.arguments`; emitting early yields empty `{}` and breaks tools (e.g. shell).
#[inline]
fn finish_reason_indicates_completed_tool_calls(finish_reason: Option<&str>) -> bool {
    finish_reason.is_some_and(|reason| {
        reason.eq_ignore_ascii_case("tool_calls") || reason.eq_ignore_ascii_case("function_call")
    })
}

/// Process a single parsed SSE chunk: emit text/reasoning deltas, accumulate
/// tool calls, and emit completed tool calls.  Returns `true` if the caller
/// should return early (tx send failed).
async fn process_sse_chunk(
    chunk: &StreamChunkResponse,
    tool_calls: &mut Vec<StreamToolCallAccumulator>,
    emitted_tool_calls: &mut bool,
    tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
) -> bool {
    let mut should_emit_tool_calls = false;
    for choice in &chunk.choices {
        if let Some(reasoning_chunk) = reasoning_stream_chunk(choice)
            && tx
                .send(Ok(StreamEvent::TextDelta(reasoning_chunk)))
                .await
                .is_err()
        {
            return true;
        }
        if let Some(ref content) = choice.delta.content
            && !content.is_empty()
            && tx
                .send(Ok(StreamEvent::TextDelta(StreamChunk {
                    delta: content.clone(),
                    reasoning: None,
                })))
                .await
                .is_err()
        {
            return true;
        }
        if let Some(details) = choice
            .delta
            .reasoning_details
            .clone()
            .filter(|v| !v.is_null())
            && tx
                .send(Ok(StreamEvent::TextDelta(StreamChunk {
                    delta: String::new(),
                    reasoning: Some(Reasoning {
                        reasoning: None,
                        reasoning_content: None,
                        reasoning_details: Some(details),
                    }),
                })))
                .await
                .is_err()
        {
            return true;
        }
        if let Some(tc_deltas) = &choice.delta.tool_calls
            && !tc_deltas.is_empty()
        {
            for tc_delta in tc_deltas {
                let index = tc_delta.index.unwrap_or(tool_calls.len());
                if index >= tool_calls.len() {
                    tool_calls.resize_with(index + 1, Default::default);
                }
                if let Some(acc) = tool_calls.get_mut(index) {
                    acc.apply_delta(tc_delta);
                }
            }
        }
        if finish_reason_indicates_completed_tool_calls(choice.finish_reason.as_deref()) {
            should_emit_tool_calls = true;
        }
    }
    if should_emit_tool_calls && !*emitted_tool_calls {
        *emitted_tool_calls = true;
        if drain_and_emit_tool_calls(tool_calls, tx).await {
            return true;
        }
    }
    false
}

/// What to do when a line in the SSE buffer fails to parse.
enum OnParseError {
    /// Send the error to the channel and abort draining.
    Abort,
    /// Skip the unparseable line and continue draining.
    Continue,
}

/// Drain accumulated tool call accumulators and emit `ToolCall` events through
/// the channel. Returns `true` if the channel receiver has been dropped
/// (caller should abort).
async fn drain_and_emit_tool_calls(
    tool_calls: &mut Vec<StreamToolCallAccumulator>,
    tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
) -> bool {
    for tc in tool_calls
        .drain(..)
        .filter_map(StreamToolCallAccumulator::into_provider_tool_call)
    {
        if tx.send(Ok(StreamEvent::ToolCall(tc))).await.is_err() {
            return true;
        }
    }
    false
}

/// Drain complete SSE lines from `buffer`, emitting text deltas and
/// accumulating tool calls.  Returns `true` when the caller should abort
/// (either the channel is closed or an `Abort`-mode parse error occurred).
async fn drain_sse_lines(
    buffer: &mut String,
    tool_calls: &mut Vec<StreamToolCallAccumulator>,
    emitted_tool_calls: &mut bool,
    tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    on_parse_error: OnParseError,
) -> bool {
    while let Some(pos) = buffer.find('\n') {
        let line = buffer[..pos].to_string();
        buffer.drain(..=pos);

        let chunk = match parse_sse_chunk(&line) {
            Ok(Some(chunk)) => chunk,
            Ok(None) => continue,
            Err(e) => match on_parse_error {
                OnParseError::Abort => {
                    let _ = tx.send(Err(e)).await;
                    return true;
                }
                OnParseError::Continue => continue,
            },
        };

        if process_sse_chunk(&chunk, tool_calls, emitted_tool_calls, tx).await {
            return true;
        }
    }
    false
}

/// Parse SSE bytes from an HTTP response and drain structured streaming events
/// directly into the provided channel — no intermediate channels or proxy loops needed.
pub(crate) async fn drain_sse_into_channel(
    response: reqwest::Response,
    tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
) {
    let mut buffer = String::new();
    let mut tool_calls: Vec<StreamToolCallAccumulator> = Vec::new();
    let mut emitted_tool_calls = false;

    debug_assert!(response.status().is_success());

    let mut bytes_stream = response.bytes_stream();
    // Accumulate partial UTF-8 sequences split across chunk boundaries.
    let mut utf8_buf: Vec<u8> = Vec::new();
    loop {
        let next = match crate::shutdown::race_shutdown(bytes_stream.next()).await {
            Ok(Some(item)) => item,
            Ok(None) => break,
            Err(_) => return,
        };
        match next {
            Ok(bytes) => {
                utf8_buf.extend_from_slice(&bytes);
                let text = match std::str::from_utf8(&utf8_buf) {
                    Ok(s) => {
                        let owned = s.to_string();
                        utf8_buf.clear();
                        owned
                    }
                    Err(e) => {
                        let valid_up_to = e.valid_up_to();
                        if valid_up_to == 0 && utf8_buf.len() < 4 {
                            continue;
                        }
                        let valid = String::from_utf8_lossy(&utf8_buf[..valid_up_to]).into_owned();
                        utf8_buf.drain(..valid_up_to);
                        valid
                    }
                };
                if text.is_empty() {
                    continue;
                }

                buffer.push_str(&text);

                if drain_sse_lines(
                    &mut buffer,
                    &mut tool_calls,
                    &mut emitted_tool_calls,
                    tx,
                    OnParseError::Abort,
                )
                .await
                {
                    return;
                }
            }
            Err(e) => {
                // ── Flush buffered SSE lines before propagating error ──
                // The byte stream may error after delivering complete
                // data (e.g. TCP RST after the last chunk).  Drain the
                // buffer so valid content isn't discarded.
                let _ = drain_sse_lines(
                    &mut buffer,
                    &mut tool_calls,
                    &mut emitted_tool_calls,
                    tx,
                    OnParseError::Continue,
                )
                .await;

                let _ = tx.send(Err(StreamError::Http(e.to_string()))).await;
                return;
            }
        }
    }

    if !emitted_tool_calls {
        drain_and_emit_tool_calls(&mut tool_calls, tx).await;
    }

    let _ = tx.send(Ok(StreamEvent::Final)).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sse_chunk_with_tool_call_delta() {
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"shell","arguments":"{\"command\":\"date\"}"}}]}}]}"#;
        let chunk = parse_sse_chunk(line)
            .unwrap()
            .expect("chunk should be parsed");
        let choice = chunk.choices.first().expect("choice should exist");
        let tool_calls = choice
            .delta
            .tool_calls
            .as_ref()
            .expect("tool call deltas should exist");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].index, Some(0));
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            Some("shell")
        );
    }

    #[test]
    fn stream_tool_call_accumulator_combines_deltas() {
        let mut acc = StreamToolCallAccumulator::default();
        acc.apply_delta(&StreamToolCallDelta {
            index: Some(0),
            id: Some("call_1".to_string()),
            function: Some(ApiToolCallFunction {
                name: Some("shell".to_string()),
                arguments: Some("{\"command\":\"".to_string()),
            }),
            name: None,
            arguments: None,
        });
        acc.apply_delta(&StreamToolCallDelta {
            index: Some(0),
            id: None,
            function: Some(ApiToolCallFunction {
                name: None,
                arguments: Some("date\"}".to_string()),
            }),
            name: None,
            arguments: None,
        });

        let tool_call = acc
            .into_provider_tool_call()
            .expect("accumulator should emit tool call");
        assert_eq!(tool_call.id, "call_1");
        assert_eq!(tool_call.name, "shell");
        assert_eq!(tool_call.arguments, serde_json::json!({"command":"date"}));
    }

    #[test]
    fn stream_tool_call_accumulator_keeps_interleaved_indices_separate() {
        let mut accumulators: Vec<StreamToolCallAccumulator> = Vec::new();
        for delta in [
            StreamToolCallDelta {
                index: Some(1),
                id: Some("call_2".to_string()),
                function: Some(ApiToolCallFunction {
                    name: Some("glob".to_string()),
                    arguments: Some("{\"pattern\":\"src/".to_string()),
                }),
                name: None,
                arguments: None,
            },
            StreamToolCallDelta {
                index: Some(0),
                id: Some("call_1".to_string()),
                function: Some(ApiToolCallFunction {
                    name: Some("shell".to_string()),
                    arguments: Some("{\"command\":\"date\"}".to_string()),
                }),
                name: None,
                arguments: None,
            },
            StreamToolCallDelta {
                index: Some(1),
                id: None,
                function: Some(ApiToolCallFunction {
                    name: None,
                    arguments: Some("\"*.rs\"}".to_string()),
                }),
                name: None,
                arguments: None,
            },
        ] {
            let idx = delta.index.unwrap_or(accumulators.len());
            if idx >= accumulators.len() {
                accumulators.resize_with(idx + 1, Default::default);
            }
            accumulators[idx].apply_delta(&delta);
        }

        let calls: Vec<_> = accumulators
            .into_iter()
            .filter_map(StreamToolCallAccumulator::into_provider_tool_call)
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[1].id, "call_2");
        assert_eq!(calls[1].name, "glob");
    }

    #[test]
    fn finish_reason_tool_calls_triggers_flush_gate() {
        assert!(finish_reason_indicates_completed_tool_calls(Some(
            "tool_calls"
        )));
        assert!(finish_reason_indicates_completed_tool_calls(Some(
            "TOOL_CALLS"
        )));
        assert!(finish_reason_indicates_completed_tool_calls(Some(
            "function_call"
        )));
        assert!(!finish_reason_indicates_completed_tool_calls(Some("stop")));
        assert!(!finish_reason_indicates_completed_tool_calls(None));
    }

    #[test]
    fn stream_tool_call_without_index_appends_new_slot() {
        let mut tool_calls: Vec<StreamToolCallAccumulator> = Vec::new();
        for tc_delta in [
            StreamToolCallDelta {
                index: None,
                id: Some("a".to_string()),
                function: Some(ApiToolCallFunction {
                    name: Some("shell".to_string()),
                    arguments: None,
                }),
                name: None,
                arguments: None,
            },
            StreamToolCallDelta {
                index: None,
                id: Some("b".to_string()),
                function: Some(ApiToolCallFunction {
                    name: Some("read".to_string()),
                    arguments: Some("{}".to_string()),
                }),
                name: None,
                arguments: None,
            },
        ] {
            let index = tc_delta.index.unwrap_or(tool_calls.len());
            if index >= tool_calls.len() {
                tool_calls.resize_with(index + 1, Default::default);
            }
            if let Some(acc) = tool_calls.get_mut(index) {
                acc.apply_delta(&tc_delta);
            }
        }
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(
            tool_calls[0]
                .clone()
                .into_provider_tool_call()
                .map(|c| c.name),
            Some("shell".to_string())
        );
        assert_eq!(
            tool_calls[1]
                .clone()
                .into_provider_tool_call()
                .map(|c| c.name),
            Some("read".to_string())
        );
    }
}
