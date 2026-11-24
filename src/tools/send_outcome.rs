//! `send_outcome` — what became of one command-line `send`, by the id its acknowledgement
//! printed.
//!
//! This is the one surface that answers a script's question about a handover it made from a
//! terminal. The record ([`crate::send_outcome`]) is the whole truth: its state, the answer the
//! addressed turn produced, the reason a failure gave, and the shape that answer came back in.
//!
//! One exception is read from elsewhere: a chain that is still running names the step it is on
//! ([`crate::jobs::chain_live_step`]), because the record cannot know where its work has reached.
//! When that read cannot say, the field is absent rather than wrong.
//!
//! An id no record stands under — another installation's, a round that was never a command-line
//! handover, or one whose record expired — answers `unknown`; an empty or malformed id is
//! `unknown` too, never a refusal. Only a failed store read is an error.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::Workspace;
use crate::send_outcome::State;
use crate::tools::Tool;

pub(crate) struct SendOutcomeTool;

#[async_trait]
impl Tool for SendOutcomeTool {
    fn name(&self) -> &'static str {
        "send_outcome"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "id": {
                    "type": "string",
                    "description": "The id a command-line `send` acknowledgement printed",
                }
            }),
            &["id"],
        )
    }

    fn side_effects(&self) -> bool {
        false // read-only record lookup
    }

    /// The body is one JSON document: the ~5 KB output budget would cut it mid-document, and a
    /// caller reading it needs the whole record.
    fn preserve_full_output(&self) -> bool {
        true
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let id = super::get_opt_str(&args, "id").unwrap_or_default();
        // An absent, empty or malformed id is `unknown`, not a refusal: there is no record to
        // read, and touching the store for it would prove nothing.
        if id.trim().is_empty() {
            return Ok(json!({ "id": id, "state": "unknown" }).to_string());
        }
        let Some(record) = crate::send_outcome::lookup(id).await? else {
            return Ok(json!({ "id": id, "state": "unknown" }).to_string());
        };
        let mut body = serde_json::Map::new();
        body.insert("id".to_string(), json!(id));
        body.insert("state".to_string(), json!(record.state.as_str()));
        if let Some(answer) = record.answer {
            body.insert("answer".to_string(), json!(answer));
        }
        if let Some(reason) = record.reason {
            body.insert("reason".to_string(), json!(reason));
        }
        if let Some(shape) = record.shape {
            body.insert("shape".to_string(), json!(shape.as_str()));
        }
        // A running chain is the one thing the record cannot know: its step comes from the
        // chain's own live state, and is left out when that read cannot say.
        if record.state == State::Running
            && record.chain_steps.is_some()
            && let Some(step) = crate::jobs::chain_live_step(id).await
        {
            body.insert("step".to_string(), json!(step));
        }
        Ok(serde_json::Value::Object(body).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> Workspace {
        crate::workspace::test_ws("send_outcome_test")
    }

    fn body(answer: &str) -> serde_json::Value {
        serde_json::from_str(answer).expect("the answer is one line of JSON")
    }

    /// An id no record stands under answers `unknown`, and nothing else.
    #[tokio::test]
    async fn an_unknown_id_answers_unknown() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        let answer = SendOutcomeTool
            .execute(&ws(), json!({ "id": "no_such_handover" }))
            .await
            .expect("the lookup answers");
        assert_eq!(
            answer,
            json!({ "id": "no_such_handover", "state": "unknown" }).to_string()
        );
    }

    /// A finished record's body carries the answer and the shape it came back in.
    #[tokio::test]
    async fn a_finished_record_carries_its_answer_and_shape() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        crate::send_outcome::begin("tool_done", None).await;
        crate::send_outcome::finish("tool_done", "the answer", crate::send_outcome::Shape::Json)
            .await;
        let parsed = body(
            &SendOutcomeTool
                .execute(&ws(), json!({ "id": "tool_done" }))
                .await
                .expect("the lookup answers"),
        );
        assert_eq!(parsed["id"], "tool_done");
        assert_eq!(parsed["state"], "done");
        assert_eq!(parsed["answer"], "the answer");
        assert_eq!(parsed["shape"], "json");
        assert!(parsed.get("reason").is_none());
        assert!(parsed.get("step").is_none());
    }

    /// A failed record carries the reason it failed.
    #[tokio::test]
    async fn a_failed_record_carries_its_reason() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        crate::send_outcome::begin("tool_failed", None).await;
        crate::send_outcome::fail("tool_failed", "stopped from the dashboard").await;
        let parsed = body(
            &SendOutcomeTool
                .execute(&ws(), json!({ "id": "tool_failed" }))
                .await
                .expect("the lookup answers"),
        );
        assert_eq!(parsed["state"], "failed");
        assert_eq!(parsed["reason"], "stopped from the dashboard");
        assert!(parsed.get("answer").is_none());
    }

    /// An empty id answers `unknown` without touching the store.
    #[tokio::test]
    async fn an_empty_id_answers_unknown() {
        let answer = SendOutcomeTool
            .execute(&ws(), json!({ "id": "   " }))
            .await
            .expect("the lookup answers");
        assert_eq!(
            answer,
            json!({ "id": "   ", "state": "unknown" }).to_string()
        );
    }

    /// A running chain record names the step its own live record is on.
    #[tokio::test]
    async fn a_running_chain_record_carries_its_step() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        crate::send_outcome::begin("tool_chain", Some(3)).await;
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            crate::launch::record_id("tool_chain", 2),
            "worker",
            "worker",
            "tool_chain_ws",
        )
        .timestamps(crate::db::now())
        .insert()
        .await
        .expect("seed the chain's live step");

        let parsed = body(
            &SendOutcomeTool
                .execute(&ws(), json!({ "id": "tool_chain" }))
                .await
                .expect("the lookup answers"),
        );
        assert_eq!(parsed["state"], "running");
        assert_eq!(parsed["step"], 2);
    }

    /// When no read can name the chain's step, the field is absent rather than wrong.
    #[tokio::test]
    async fn a_running_chain_record_omits_a_step_it_cannot_read() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        crate::send_outcome::begin("tool_chain_gone", Some(2)).await;
        let parsed = body(
            &SendOutcomeTool
                .execute(&ws(), json!({ "id": "tool_chain_gone" }))
                .await
                .expect("the lookup answers"),
        );
        assert_eq!(parsed["state"], "running");
        assert!(parsed.get("step").is_none());
    }
}
