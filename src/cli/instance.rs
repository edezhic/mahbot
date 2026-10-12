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
    // nothing. The command acts as the machine's owner. No agent id and no
    // channel are set: there is no agent session behind the command and nothing
    // to reply to.
    let user_name = crate::users::ADMIN_USER_NAME.to_string();
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
    let outcome = crate::agent::CURRENT_TOOL_USER_NAME
        .scope(user_name, async {
            crate::agent::CURRENT_TOOL_AGENT_TRACKING
                .scope(tracking, async {
                    match AssertUnwindSafe(tool.execute_answer(&ws, args))
                        .catch_unwind()
                        .await
                    {
                        Ok(Ok(answer)) => Ok(answer),
                        // A panic is a refusal carrying the panic's own
                        // message, never a severed exchange — the crate builds
                        // with `panic = "unwind"` on purpose.
                        Err(payload) => Err(crate::util::panic_message(&*payload)),
                        // The tool's own wording, unwrapped: the caller reads
                        // it as the product's answer, so it must not carry the
                        // `Error executing <tool>: …` envelope that is worded
                        // for a model's failure block. The record below wraps
                        // it the way an agent's failure is recorded.
                        Ok(Err(e)) => Err(e.to_string()),
                    }
                })
                .await
        })
        .instrument(span)
        .await;
    let duration_ms = crate::util::millis_i64(started.elapsed());

    record_operation(&call, tool_name, duration_ms, &outcome, &ws.path).await;

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

/// The refusal for a tool this command line does not front — a name no family
/// publishes, or one the owner's toolset does not hold. The caller's own secrets
/// are removed even here: the tool name is caller input like any other.
fn unknown_tool_refusal(call: &CallRequest) -> CallReply {
    CallReply::refused(crate::util::redact_secrets(
        &format!(
            "no tool named '{}' is available on the command line — it fronts the tools its own \
             commands expose and nothing else",
            call.tool
        ),
        &call.secrets,
    ))
}

/// Record the operation exactly as the agent path records a call, with
/// [`CallSource::CommandLine`](crate::stats::CallSource::CommandLine).
///
/// `agent_id`/`role`/`source` are what make a change a script made
/// distinguishable from one the assistant made — the row is otherwise the same
/// shape, in the same table, attributed to the owner's workspace path. The
/// caller's own secrets are removed from the recorded error text as well as
/// from the arguments: a failed tool can quote what it was given. The reason is
/// recorded with the same `Error executing <tool>: …` wrap the agent path uses
/// ([`crate::tools::tool_failure_reason`]), even though the caller was answered
/// with the tool's own words — the record is read as a tool failure, not as a
/// human's answer.
async fn record_operation(
    call: &CallRequest,
    tool_name: &str,
    duration_ms: i64,
    outcome: &Result<ToolAnswer, String>,
    workspace_path: &str,
) {
    let record = crate::ToolCallRecord {
        tool_name: tool_name.to_string(),
        // The caller's own secrets are removed before the serialized arguments
        // are scrubbed and capped.
        arguments: crate::stats::recorded_arguments(&call.args, &call.secrets),
        duration_ms,
        success: outcome.is_ok(),
        error_message: match outcome {
            Ok(_) => None,
            Err(reason) => {
                let reason = crate::util::redact_secrets(reason, &call.secrets);
                let reason = crate::tools::tool_failure_reason(tool_name, &reason);
                (!reason.is_empty()).then_some(reason)
            }
        },
    };
    // An absent log store is not an error: the operation's own answer is what
    // the caller is owed, and the record is a side benefit of it.
    if let Some(store) = crate::logs::LOG_STORE.get()
        && let Err(e) = store
            .flush_batch(
                "cli",
                "cli",
                workspace_path,
                crate::stats::CallSource::CommandLine,
                std::slice::from_ref(&record),
            )
            .await
    {
        tracing::warn!(
            error = %e,
            tool = %tool_name,
            "cli: failed to record the command-line operation"
        );
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
        })
        .await;
        assert_eq!(note.outcome, CallOutcome::Done, "answer: {}", note.answer);
        assert!(
            note.answer.contains(&id) && note.answer.contains("assistant"),
            "a note is recorded as the assistant leaves it: {}",
            note.answer
        );
    }
}
