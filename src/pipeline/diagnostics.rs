//! InDiagnostics phase module — runs discovered diagnostics commands.

use std::fmt::Write;
use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::load_prompt;
use crate::tools::shell::{ShellMode, ShellTool};
use crate::{DiagnosticsCommands, Workspace};

use super::{
    DIAGNOSTICS_ROLE, TicketPhase, TransitionCtx, bounce_to_development,
    comment_and_transition_or_bail, guard_job_phase, reset_phase_attempt, warn,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, TicketPhase::InDiagnostics, &job_id).await {
        return;
    }
    dispatch_diagnostics(ticket, ws, &job_id).await;
}

/// Run diagnostics commands sequentially, collecting output and pass/fail status.
///
/// The judgement is the exit status and nothing else (`Some(0)` is a pass): what a
/// command writes is kept for the comment — through the output profiles, which
/// trim it for display — but never read to decide the outcome. A check command
/// answering in another language is therefore judged exactly as it is in English
/// (see `profiles`' own note).
async fn run_diagnostics_commands(diag: &DiagnosticsCommands, ws: &Workspace) -> (String, bool) {
    let mut comment = String::new();
    let mut all_passed = true;
    let mut failed_at: &str = "";

    for (label, cmd_opt) in diag.commands() {
        let Some(cmd) = cmd_opt else {
            continue;
        };

        let mut mark_failed = |comment: &mut String, body: String| {
            let _ = write!(comment, "\n\n{label} ({cmd}):\n");
            comment.push_str(&body);
            all_passed = false;
            failed_at = label;
        };

        let started = std::time::Instant::now();
        match ShellTool::new(ShellMode::Full)
            .execute_with_status(ws, serde_json::json!({"command": cmd}))
            .await
        {
            Ok((_output, Some(0))) => {
                let _ = write!(
                    comment,
                    "\n\n{label} ({cmd}): PASSED in {:.1}s",
                    started.elapsed().as_secs_f64(),
                );
            }
            Ok((output, _exit_code)) => {
                let display = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                mark_failed(&mut comment, display);
                break;
            }
            Err(e) => {
                mark_failed(&mut comment, e.to_string());
                break;
            }
        }
    }

    if all_passed {
        comment.push_str("\n\n---\n");
        comment.push_str(&load_prompt("pipeline/diagnostics_passed.md"));
    } else {
        let _ = write!(
            comment,
            "\n\n---\n{} {failed_at}",
            load_prompt("pipeline/diagnostics_failed.md"),
        );
    }

    crate::tools::shell::cleanup_agent_spills(DIAGNOSTICS_ROLE);
    (comment.trim_start_matches('\n').to_string(), all_passed)
}

/// Conclude a successful diagnostics run — transition to InReview and delete
/// the phase job (the puller creates the InReview job).
async fn conclude_diagnostics_success(
    ticket: &Ticket,
    job_id: &str,
    comment: &str,
    log_label: &str,
) {
    comment_and_transition_or_bail(
        TransitionCtx::buffered(
            ticket,
            TicketPhase::InDiagnostics,
            TicketPhase::InReview,
            "Diagnostics",
            DIAGNOSTICS_ROLE,
        ),
        DIAGNOSTICS_ROLE,
        comment,
        log_label,
    )
    .await;
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

/// Conclude a failed diagnostics run — unified bounce back to development.
async fn conclude_diagnostics_failure(ticket: &Ticket, job_id: &str, comment: &str) {
    bounce_to_development(
        ticket,
        TicketPhase::InDiagnostics,
        "Diagnostics",
        DIAGNOSTICS_ROLE,
        DIAGNOSTICS_ROLE,
        comment,
        job_id,
    )
    .await;
}

/// Run diagnostics commands after the engineer completes development.
async fn dispatch_diagnostics(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    // Register a synthetic in-flight roster marker so the phase job's
    // re-dispatch guard blocks while diagnostics execute. Cleared when the
    // job is deleted at phase completion.
    let diag_agent_id = format!("ticket_{}_diagnostics", ticket.id);
    if let Err(e) = crate::jobs::upsert_job_agent(
        &crate::session::store().conn,
        job_id,
        &diag_agent_id,
        crate::jobs::AgentKind::Diagnostics,
        crate::jobs::RowStatus::Launched,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = ?e,
            "Failed to register diagnostics in-flight marker — diagnostics may re-dispatch",
        );
    }

    match crate::workspace::store().get_diagnostics(&ws.name).await {
        Ok(Some(cmds)) if !cmds.is_empty() => {
            let (comment, all_passed) = run_diagnostics_commands(&cmds, &ws).await;

            if guard_job_phase(&ticket.id, TicketPhase::InDiagnostics, job_id).await {
                return;
            }

            if all_passed {
                conclude_diagnostics_success(
                    &ticket,
                    job_id,
                    &comment,
                    "Diagnostics finished — transitioned ticket",
                )
                .await;
            } else {
                conclude_diagnostics_failure(&ticket, job_id, &comment).await;
            }
        }
        Ok(_) => {
            conclude_diagnostics_success(
                &ticket,
                job_id,
                "No diagnostics commands are configured for this workspace \
                 — diagnostics skipped.",
                "Diagnostics skipped — transitioned ticket",
            )
            .await;
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to load diagnostics for workspace — resetting for a fresh attempt",
            );
            reset_phase_attempt(
                &ticket,
                TicketPhase::InDiagnostics,
                job_id,
                "diagnostics load failure",
                &format!("Could not load diagnostics commands due to a database error: {e}"),
            )
            .await;
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// The locales to try a translating answer under.
    const LOCALE_CANDIDATES: &[&str] = &["de_DE.UTF-8", "fr_FR.UTF-8", "es_ES.UTF-8"];

    /// The failing command the lane judges: a nested shell formats a
    /// locale-dependent word and exits non-zero, so the observation needs nothing
    /// installed beyond the shell every unix host has.
    const COMMAND: &str = "sh -c 'printf \"%s\\n\" \"$(date +%A)\"; exit 1'";

    /// Run one command through the product's own command path — the same path the
    /// judgement below uses — and report what it wrote and its exit status. `None`
    /// when the tool refused to run it at all (which is no observation).
    async fn run_command(ws: &Workspace, command: &str) -> Option<(String, Option<i32>)> {
        ShellTool::new(ShellMode::Full)
            .execute_with_status(ws, serde_json::json!({ "command": command }))
            .await
            .ok()
    }

    /// The first locale this host answers the command in another language under;
    /// `None` when none of the candidates is installed here.
    async fn pick_translating_locale(ws: &Workspace) -> Option<&'static str> {
        let (plain, _) = run_command(ws, &format!("LC_ALL=C {COMMAND}")).await?;
        for locale in LOCALE_CANDIDATES {
            let Some((translated, _)) =
                run_command(ws, &format!("LC_ALL={locale} {COMMAND}")).await
            else {
                continue;
            };
            if translated != plain {
                return Some(locale);
            }
        }
        None
    }

    /// What the product's own judgement of a check command's result does when the
    /// owner's locale makes that command answer in another language.
    ///
    /// The instrument is [`run_diagnostics_commands`] itself: the same failing
    /// command is judged under `C` and under a locale this host translates it
    /// with. Only the verdict, the exit status and the shape of what was shown are
    /// printed — never the text. The lane is unix-only (the module itself is): it
    /// leans on the `LC_ALL=<locale>` prefix the shell applies to the command it is
    /// given. Run it with
    ///
    /// ```text
    /// cargo test --lib -- --ignored --nocapture judgement_never_reads_the_answers_language
    /// ```
    #[tokio::test]
    #[ignore = "manual evidence run: needs a locale that translates a failing command's answer"]
    async fn judgement_never_reads_the_answers_language() {
        /// One run under `label`, as the row reports it: what the command wrote, its
        /// exit status, the product's verdict and how much was shown. The command
        /// runs twice on purpose — [`run_diagnostics_commands`] returns the verdict
        /// and what it showed, never the status it judged on.
        async fn row(ws: &Workspace, label: &str) -> (String, Option<i32>, bool, usize) {
            let line = format!("LC_ALL={label} {COMMAND}");
            let (answer, status) = run_command(ws, &line).await.expect("the command runs");
            let diag = DiagnosticsCommands {
                format: Some(line),
                ..Default::default()
            };
            let (comment, passed) = run_diagnostics_commands(&diag, ws).await;
            (answer, status, passed, comment.lines().count())
        }

        let tmp = TempDir::new().expect("tempdir");
        let ws = crate::workspace::test_ws(tmp.path());

        let Some(locale) = pick_translating_locale(&ws).await else {
            println!("EVIDENCE SKIPPED: no candidate locale translates the answer on this host");
            return;
        };
        println!("command: <the failing command> | locale: {locale}");

        let (c_answer, c_status, c_passed, c_shown) = row(&ws, "C").await;
        println!(
            "locale C: judged {} | exit status {c_status:?} | shown {c_shown} lines",
            if c_passed { "passed" } else { "failed" },
        );
        let (answer, status, passed, shown) = row(&ws, locale).await;
        println!(
            "locale {locale}: judged {} | exit status {status:?} | shown {shown} lines | answer differs from the C run: {}",
            if passed { "passed" } else { "failed" },
            c_answer != answer,
        );
        let identical = c_passed == passed;
        println!("verdict identical across locales: {identical}");
        println!("the verdict is the exit status, never the text");
        assert!(
            identical,
            "the judgement must not depend on the language the command answers in"
        );
    }
}
