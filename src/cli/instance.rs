//! Carrying one command-line operation out in the running product.
//!
//! This is the instance side of the command line: the endpoint hands over a
//! [`CallRequest`] and gets back a [`CallReply`]. The operation runs on the
//! very tool an agent's assistant calls, in the owner's own workspace and with
//! the owner's own toolset — so the frontend has no rules of its own to keep in
//! step with the product's.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Instant;

use futures_util::FutureExt as _;
use tracing::Instrument as _;

use crate::db::ipc::{CallReply, CallRequest};
use crate::tools::ToolAnswer;

/// Carry one command-line operation out in the running product.
///
/// An operation is a side effect that may route into the message pipeline, so
/// it is never served before the instance has finished starting
/// ([`crate::cli::mark_ready`]) and never on a socket another user could reach
/// (handled by the endpoint). The endpoint also holds the registry guard that
/// keeps a graceful drain from ending while this is in flight, across the
/// writing of the answer too — see [`hold_flight`].
pub(crate) async fn execute(call: CallRequest) -> CallReply {
    match crate::cli::wait_for_service().await {
        crate::cli::ServiceGate::Ready => {}
        crate::cli::ServiceGate::Unavailable => {
            return CallReply::unavailable(
                "the instance did not finish starting up, so it cannot serve command-line \
                 operations",
            );
        }
        crate::cli::ServiceGate::StillStarting => {
            return CallReply::unavailable(
                "the instance has not finished starting up, so it cannot serve command-line \
                 operations yet — nothing ran; try again once it is up",
            );
        }
        crate::cli::ServiceGate::NotAccepted => {
            return CallReply::not_accepted(
                "the instance is shutting down, so it did not take the operation on — nothing \
                 ran; try again once it is up",
            );
        }
    }

    // Only a tool a frontend publishes is served by name: the owner's whole
    // toolset must not be reachable through the channel, or the surface a
    // same-user socket can drive would be neither reviewable nor tied to the
    // command line. The caller's own secrets are removed even from this refusal:
    // the tool name is caller input like any other.
    if !crate::cli::fronts_tool(&call.tool) {
        return unknown_tool_refusal(&call);
    }

    // The owner's workspace and the admin Assistant's own toolset: the command
    // line exposes exactly what the owner's own tools hold and nothing else.
    let ws = crate::users::personal_workspace_struct(crate::users::ADMIN_USER_NAME);
    let tools = crate::Role::Assistant.tools(
        &ws,
        true,
        Arc::new(crate::tools::chrome::ChromeRunSessions::default()),
    );
    let Some(tool) = crate::tools::find_tool(&tools, &call.tool) else {
        // Unreachable while the registry test holds — every fronted tool is in
        // this toolset — but a refusal, never a panic, if the two ever drift.
        return unknown_tool_refusal(&call);
    };
    let tool_name = tool.name();

    // Everything the tool's task-locals need is read SYNCHRONOUSLY, before the
    // await — the task-local contract: a read inside the awaited tool would see
    // nothing. The command acts as the machine's owner. No agent id is set:
    // there is no agent session behind the command. The channel IS set to the
    // command line's own: it attributes the operation to the shell rather than
    // to a dashboard the owner typed into, and it travels with a message the
    // command line hands over. The MARKER is what makes the call recognisable
    // as the command line's — it is the flag tool code reads to behave as the
    // shell caller's (see `SendTool::command_line`), while the channel is set
    // for provenance and cannot stand in for it.
    let user_name = crate::users::ADMIN_USER_NAME.to_string();
    let channel = crate::cli::CHANNEL.to_string();
    let tracking = Some(crate::agent::registry::AgentTracking {
        agent_id: "cli".to_string(),
        generation: 0,
        role: "cli".to_string(),
        workspace: ws.name.clone(),
    });
    // The same span the agent loop opens, so the logs bridge attributes the
    // operation's rows to `cli` exactly as it attributes an agent's.
    let span = tracing::info_span!(
        "agent",
        agent_id = %"cli",
        role = %"cli",
        workspace = %ws.path
    );

    let args = call.args.clone();
    let started = Instant::now();
    // The destination a command line chose for the files this call produces, as
    // the producing tools read it through [`crate::agent::tool_output_dir`]. It
    // is read here because nothing in the tools' arguments carries it — a tool
    // argument would be model input, while this is the caller's own operand.
    let output_dir = call.output_dir.as_deref().map(std::path::PathBuf::from);
    let outcome = crate::agent::CURRENT_TOOL_COMMAND_LINE
        .scope(true, async {
            crate::agent::CURRENT_TOOL_USER_NAME
                .scope(user_name, async {
                    crate::agent::CURRENT_TOOL_CHANNEL
                        .scope(channel, async {
                            crate::agent::CURRENT_TOOL_AGENT_TRACKING
                                .scope(tracking, async {
                                    crate::agent::CURRENT_TOOL_OUTPUT_DIR
                                        .scope(output_dir, async {
                                            match AssertUnwindSafe(tool.execute_answer(&ws, args))
                                                .catch_unwind()
                                                .await
                                            {
                                                Ok(Ok(answer)) => Ok(answer),
                                                // A panic is a refusal carrying the panic's own
                                                // message, never a severed exchange — the crate builds
                                                // with `panic = "unwind"` on purpose.
                                                Err(payload) => {
                                                    Err(crate::util::panic_message(&*payload))
                                                }
                                                // The tool's own wording, unwrapped: the caller reads
                                                // it as the product's answer, so it must not carry the
                                                // `Error executing <tool>: …` envelope that is worded
                                                // for a model's failure block. The record below wraps
                                                // it the way an agent's failure is recorded. Any
                                                // context chain is kept — `Display` alone would hide
                                                // the cause the outermost context names.
                                                Ok(Err(e)) => Err(format!("{e:#}")),
                                            }
                                        })
                                        .await
                                })
                                .await
                        })
                        .await
                })
                .await
        })
        .instrument(span)
        .await;
    let duration_ms = crate::util::millis_i64(started.elapsed());

    record_operation(&call, tool_name, duration_ms, &outcome, &ws.path);

    reply(outcome, &call.secrets)
}

/// Hold the registry guard that keeps a graceful drain (or an update hand-off)
/// from ending while one command-line operation is in flight.
///
/// The endpoint takes this BEFORE the operation is served and holds it until the
/// answer has been written, so an operation the product accepted is always
/// finished and always reported — there is no window in which it can be carried
/// out yet lose its answer. The drain-watch fires the shutdown token only when
/// `AGENT_REGISTRY` and `NON_AGENT_CALLS` are BOTH empty.
pub(crate) fn hold_flight() -> crate::agent::registry::NonAgentCallGuard {
    let ws = crate::users::personal_workspace_struct(crate::users::ADMIN_USER_NAME);
    crate::agent::registry::NON_AGENT_CALLS.register("cli_command", &ws.name, None, false, None)
}

/// The answer for a tool this command line does not serve — a name no family
/// publishes, or one the owner's toolset does not hold. The running product
/// cannot take this operation on (one shared command-line contract: see
/// [`crate::cli::FAMILIES`]), so it is [`Unavailable`](CallReply::unavailable),
/// NOT `Refused`.
///
/// The two callers are not the same kind of thing. A name no family fronts
/// ([`crate::cli::fronts_tool`]) is the safety net for a frontend whose family
/// this build does not know while the value it sends is one this build serves —
/// a build that added a family without bumping that value; the reading side
/// treats this class as the environment's, so the sentence carries the restart
/// advice. A tool a family DOES front that the owner's toolset does not hold is a
/// product-side inconsistency ([`crate::Role::Assistant`]'s own toolset is the
/// command line's whole surface), where nothing is implied about the build and a
/// restart is the wrong advice. The caller's own secrets are removed on both
/// paths: the tool name is caller input like any other.
fn unknown_tool_refusal(call: &CallRequest) -> CallReply {
    let wording = if crate::cli::fronts_tool(&call.tool) {
        format!(
            "the running product's command line holds no tool named '{}' — nothing ran",
            call.tool
        )
    } else {
        format!(
            "the running product does not serve '{}' on its command line — it was built before \
             that command existed, so restart it from this build and try again; nothing ran",
            call.tool
        )
    };
    CallReply::unavailable(crate::util::redact_secrets(&wording, &call.secrets))
}

/// What a partly carried-out operation's row says: the tool's own text names the
/// part that did not happen, and this is the reason a reader sees beside the
/// unset `success`.
const PARTIAL_RECORD_REASON: &str = "the operation reported that part of it was not done";

/// Record the operation exactly as the agent path records a call, with
/// [`CallSource::CommandLine`](crate::stats::CallSource::CommandLine).
///
/// `agent_id`/`role`/`source` are what make a change a script made
/// distinguishable from one the assistant made — the row is otherwise the same
/// shape, in the same table, attributed to the owner's workspace path.
fn record_operation(
    call: &CallRequest,
    tool_name: &str,
    duration_ms: i64,
    outcome: &Result<ToolAnswer, String>,
    workspace_path: &str,
) {
    let record = operation_record(call, tool_name, duration_ms, outcome);
    // The row goes on the telemetry queue like the agent path's: the operation's
    // own answer is what the caller is owed.
    crate::telemetry::enqueue(crate::telemetry::TelemetryRecord::ToolCalls(
        crate::telemetry::ToolCallBatch {
            source: crate::stats::CallSource::CommandLine,
            agent_id: "cli".to_string(),
            role: "cli".to_string(),
            workspace: workspace_path.to_string(),
            recorded_at: crate::db::now(),
            records: vec![record],
        },
    ));
}

/// The row a finished operation becomes.
///
/// The caller's own secrets are removed from the recorded error text as well as
/// from the arguments: a failed tool can quote what it was given. The reason is
/// recorded with the same `Error executing <tool>: …` wrap the agent path uses
/// ([`crate::tools::tool_failure_reason`]), even though the caller was answered
/// with the tool's own words — the record is read as a tool failure, not as a
/// human's answer. A partly carried-out operation is recorded as
/// [`PARTIAL_RECORD_REASON`] with `success` unset: the caller is answered with a
/// failed operation, and the row is what a reader filters on, so the two agree.
fn operation_record(
    call: &CallRequest,
    tool_name: &str,
    duration_ms: i64,
    outcome: &Result<ToolAnswer, String>,
) -> crate::ToolCallRecord {
    crate::ToolCallRecord {
        tool_name: tool_name.to_string(),
        // The caller's own secrets are removed before the serialized arguments
        // are scrubbed and capped. The DESTINATION is deliberately not among
        // them: it is this command line's own operand rather than an argument the
        // tool was handed, and the answer already names every produced file by
        // its absolute path.
        arguments: crate::stats::recorded_arguments(&call.args, &call.secrets),
        duration_ms,
        success: matches!(outcome, Ok(answer) if !answer.partial),
        error_message: match outcome {
            Ok(answer) if answer.partial => Some(PARTIAL_RECORD_REASON.to_string()),
            Ok(_) => None,
            Err(reason) => {
                let reason = crate::util::redact_secrets(reason, &call.secrets);
                let reason = crate::tools::tool_failure_reason(tool_name, &reason);
                (!reason.is_empty()).then_some(reason)
            }
        },
    }
}

/// The reply one finished execution becomes.
///
/// Any text that goes back to the caller — success or failure — passes through
/// [`crate::util::redact_secrets`], so no value the caller handed over as a
/// secret can reach its own answer. The product's answer is otherwise neither
/// shortened nor rewritten: the caller is owed what the tool said, and a
/// partly carried-out operation is marked as such rather than dressed up.
fn reply(outcome: Result<ToolAnswer, String>, secrets: &[String]) -> CallReply {
    match outcome {
        Ok(answer) => {
            let text = crate::util::redact_secrets(&answer.text, secrets);
            if answer.partial {
                CallReply::partial(text)
            } else {
                CallReply::done(text)
            }
        }
        Err(reason) => CallReply::refused(crate::util::redact_secrets(&reason, secrets)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ipc::CallOutcome;

    /// The reply carries what the tool said — with the caller's own secrets
    /// removed — and a partly carried-out operation is marked `Partial` while a
    /// whole one is `Done`. A refusal is the tool's own wording, as the caller
    /// sees it: the `Error executing <tool>: …` wrap the record carries is added
    /// by [`record_operation`], never by the reply.
    #[test]
    fn replies_carry_the_answer_with_the_callers_secrets_removed() {
        let secrets = vec!["s3cret".to_string()];

        let done = reply(
            Ok(ToolAnswer::done("registered, key s3cret stored")),
            &secrets,
        );
        assert_eq!(done.outcome, CallOutcome::Done);
        assert_eq!(done.answer, "registered, key [REDACTED] stored");

        let partial = reply(
            Ok(ToolAnswer::partial(
                "registered, but the activation write failed",
            )),
            &secrets,
        );
        assert_eq!(partial.outcome, CallOutcome::Partial);

        let refused = reply(Err("no workspace named s3cret".to_string()), &secrets);
        assert_eq!(refused.outcome, CallOutcome::Refused);
        assert_eq!(refused.answer, "no workspace named [REDACTED]");
    }

    /// The row a recorded operation leaves behind says what the caller saw: a
    /// whole operation is a success with nothing to report, a partly carried-out
    /// one is a failure naming the shortfall, and a refusal keeps the tool's own
    /// reason — with the caller's secrets scrubbed out of it.
    #[test]
    fn a_recorded_row_reports_a_partial_answer_as_a_failure() {
        let call = |args: serde_json::Value, secrets: Vec<String>| CallRequest {
            tool: "document".to_string(),
            args,
            secrets,
            output_dir: None,
        };

        let done = operation_record(
            &call(serde_json::json!({}), Vec::new()),
            "document",
            5,
            &Ok(ToolAnswer::done("a document was produced")),
        );
        assert!(done.success);
        assert_eq!(done.error_message, None);

        let partial = operation_record(
            &call(serde_json::json!({}), Vec::new()),
            "document",
            5,
            &Ok(ToolAnswer::partial("produced, but a placeholder stayed")),
        );
        assert!(!partial.success);
        assert_eq!(
            partial.error_message.as_deref(),
            Some(PARTIAL_RECORD_REASON)
        );

        let refused = operation_record(
            &call(
                serde_json::json!({ "key": "s3cret" }),
                vec!["s3cret".to_string()],
            ),
            "document",
            5,
            &Err("nothing named s3cret".to_string()),
        );
        assert!(!refused.success);
        assert_eq!(
            refused.error_message.as_deref(),
            Some("Error executing document: nothing named [REDACTED]")
        );
    }

    /// End to end through the instance side: the two `workspace` operations that
    /// need a ticket to exist are served on the running product's own tool and
    /// come back as the answers the caller reads — `ticket` and `comment`
    /// (whose note is attributed to `assistant`, as every note left through that
    /// operation is).
    #[tokio::test]
    async fn execute_serves_the_workspace_ticket_and_comment_operations() {
        crate::util::test::init_test_stores().await;
        crate::util::test::create_test_workspace("/tmp/ws_cli_ops", "ws_cli_ops").await;
        let id = crate::pipeline::board::store()
            .create_ticket(&crate::pipeline::board::TicketParams {
                title: "Ticket the command line reads".to_string(),
                description: "created by the instance-side test".to_string(),
                workspace_name: "ws_cli_ops".to_string(),
                phase: crate::pipeline::board::TicketPhase::Backlog,
                prerequisites: Vec::new(),
                reporter: "cli-test".to_string(),
                embedding: None,
                priority: 1,
            })
            .await
            .expect("create the ticket the operations act on");
        // The endpoint's own gate: an operation is only served by a product that
        // finished starting (the same signal `bootstrap_mahbot` sends).
        crate::cli::mark_ready();

        let show = execute(CallRequest {
            tool: "workspace".to_string(),
            args: serde_json::json!({ "action": "ticket", "ticket_id": id }),
            secrets: Vec::new(),
            output_dir: None,
        })
        .await;
        assert_eq!(show.outcome, CallOutcome::Done, "answer: {}", show.answer);
        assert!(
            show.answer.contains("Ticket the command line reads"),
            "the ticket's own display must reach the caller: {}",
            show.answer
        );

        let note = execute(CallRequest {
            tool: "workspace".to_string(),
            args: serde_json::json!({
                "action": "comment",
                "ticket_id": id,
                "content": "a note from a script",
            }),
            secrets: Vec::new(),
            output_dir: None,
        })
        .await;
        assert_eq!(note.outcome, CallOutcome::Done, "answer: {}", note.answer);
        assert!(
            note.answer.contains(&id) && note.answer.contains("assistant"),
            "a note is recorded as the assistant leaves it: {}",
            note.answer
        );
    }

    /// End to end through the instance side for a PRODUCING family: the
    /// operation runs the owner's own document tool, and what it produces lands
    /// in the destination the command line carried beside the tool's arguments —
    /// not in the workspace directory an agent's call would use — with the
    /// answer naming it by its absolute path. The destination is the command
    /// line's own operand and never reaches the tool's args, which is why the
    /// tool's own answer stays word-for-word what an agent reads. An operation
    /// that produces SEVERAL files puts every one there too.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn execute_produces_a_document_into_the_destination_the_caller_chose() {
        /// Run one document operation into the destination and return every file
        /// the product's own answer names, each absolute and the destination's child.
        async fn drive(
            destination: &std::path::Path,
            args: serde_json::Value,
        ) -> Vec<std::path::PathBuf> {
            let reply = execute(CallRequest {
                tool: "document".to_string(),
                args,
                secrets: Vec::new(),
                output_dir: Some(destination.to_string_lossy().into_owned()),
            })
            .await;
            assert_eq!(reply.outcome, CallOutcome::Done, "answer: {}", reply.answer);
            reply
                .answer
                .split("[FILE:")
                .skip(1)
                .map(|rest| std::path::PathBuf::from(rest.split(']').next().unwrap_or_default()))
                .inspect(|file| {
                    assert!(
                        file.is_absolute() && file.exists() && file.parent() == Some(destination),
                        "a produced file is named by its absolute path, is there, and lands in \
                         the destination the caller chose: {}",
                        file.display()
                    );
                })
                .collect()
        }

        if crate::tools::bun::bun_binary_path().is_none() {
            return;
        }
        crate::util::test::init_test_stores().await;
        crate::cli::mark_ready();

        let destination = crate::util::test::test_root().join("cli-destination");
        let _ = tokio::fs::remove_dir_all(&destination).await;
        tokio::fs::create_dir_all(&destination)
            .await
            .expect("a destination of the caller's own");

        let created = drive(
            &destination,
            serde_json::json!({ "action": "create", "format": "docx", "file_name": "from-a-script", "content": [{ "type": "paragraph", "text": "produced by a script" }] }),
        )
        .await;
        assert_eq!(created, [destination.join("from-a-script.docx")]);

        // The multi-file case: two single-page PDFs merged into one, and that
        // merge split back into one file per page.
        let mut pages = Vec::new();
        for name in ["one", "two"] {
            let pdf = drive(
                &destination,
                serde_json::json!({ "action": "create", "format": "pdf", "file_name": name, "content": [{ "type": "paragraph", "text": name }] }),
            )
            .await;
            pages.push(pdf.into_iter().next().expect("create writes one pdf"));
        }
        let merged = drive(
            &destination,
            serde_json::json!({ "action": "pdf_merge", "paths": [pages[0].to_string_lossy(), pages[1].to_string_lossy()], "file_name": "merged" }),
        )
        .await;
        let merged = merged.first().expect("a merge writes one file");
        let parts = drive(
            &destination,
            serde_json::json!({ "action": "pdf_split", "path": merged.to_string_lossy(), "pages": ["1", "2"], "file_name": "page" }),
        )
        .await;
        assert_eq!(parts.len(), 2, "one file per page entry: {parts:?}");

        // The agents' own generated directory is untouched: the destination
        // replaced it for these calls alone.
        let agents = crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME)
            .join("generated")
            .join("from-a-script.docx");
        assert!(
            tokio::fs::metadata(&agents).await.is_err(),
            "a command-line call must not write into the agent's own generated directory"
        );
    }

    /// A partly carried-out document operation reaches the caller as `Partial`:
    /// the tool's own flag, never a reading of its wording — and the answer is
    /// delivered whole.
    #[tokio::test]
    #[ignore = "requires the managed bun runtime, installed on the product's first start; runs only when explicitly invoked"]
    async fn execute_reports_a_shortfall_the_document_tool_found() {
        if crate::tools::bun::bun_binary_path().is_none() {
            return;
        }
        crate::util::test::init_test_stores().await;
        crate::cli::mark_ready();

        let destination = crate::util::test::test_root().join("cli-destination-shortfall");
        let _ = tokio::fs::remove_dir_all(&destination).await;
        tokio::fs::create_dir_all(&destination)
            .await
            .expect("a destination of the caller's own");

        // A template with two placeholders of which the call fills one: the
        // fill runs, the copy is produced, and the other placeholder is left as
        // it was — a shortfall the tool's own answer names.
        let sample = destination.join("sample.docx");
        let created = execute(CallRequest {
            tool: "document".to_string(),
            args: serde_json::json!({
                "action": "create",
                "format": "docx",
                "file_name": "sample",
                "content": [{ "type": "paragraph", "text": "Hello, {name}! {unfilled}" }],
            }),
            secrets: Vec::new(),
            output_dir: Some(destination.to_string_lossy().into_owned()),
        })
        .await;
        assert_eq!(
            created.outcome,
            CallOutcome::Done,
            "answer: {}",
            created.answer
        );
        assert!(tokio::fs::metadata(&sample).await.is_ok());

        let filled = execute(CallRequest {
            tool: "document".to_string(),
            args: serde_json::json!({
                "action": "fill_template",
                "path": sample.to_string_lossy(),
                "values": { "name": "Ada" },
                "file_name": "filled",
            }),
            secrets: Vec::new(),
            output_dir: Some(destination.to_string_lossy().into_owned()),
        })
        .await;
        assert_eq!(
            filled.outcome,
            CallOutcome::Partial,
            "a shortfall must reach the caller as one: {}",
            filled.answer
        );
        assert!(
            filled.answer.contains("no values for these placeholders"),
            "the product's own answer names the part that did not succeed: {}",
            filled.answer
        );
    }
}
