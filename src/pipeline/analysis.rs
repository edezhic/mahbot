//! Analysis phase module — parallel analysts research a backlog ticket.

use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, substitute};
use crate::{Role, Workspace};

use super::{
    ANALYST_PASS_THRESHOLD, AgentSlot, BoardStore, DEFAULT_PARALLEL_AGENT_COUNT, ExtractionMode,
    FinalizeOutcome, ParallelVerdict, TicketPhase, TransitionCtx, Write, build_agent_slots,
    build_round_joint_comment, guard_job_phase, info, insert_round_slots, is_ticket_in_phase,
    load_ticket_analysis_angles, pause_freezing, reset_phase_attempt, run_parallel_agents,
    stage_name, warn, with_comment_and_transition,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String, _resumed: bool) {
    if !is_ticket_in_phase(&ticket.id, TicketPhase::Analysis).await {
        let _ = crate::jobs::complete_ticket_job(&crate::session::store().conn, &job_id).await;
        return;
    }
    dispatch_backlog_analysts(ticket, ws, &job_id).await;
}

/// Build the base analysis slot roster and write it onto the phase job.
async fn ensure_analysis_slots(
    ticket: &Ticket,
    job_id: &str,
    prompt: &str,
    count: usize,
) -> Vec<AgentSlot> {
    let slots = build_agent_slots(
        &ticket.id,
        Role::Analyst,
        prompt,
        &load_ticket_analysis_angles(),
        count,
        0,
        count,
    );
    insert_round_slots(job_id, &slots, crate::jobs::AgentKind::Analyst).await;
    slots
}

/// Append escalation slots (3, 4) to an existing phase job. Escalation slots
/// are purely blocker-verification focused and inherit NO base angle.
async fn append_analysis_slots(
    ticket: &Ticket,
    job_id: &str,
    prompt: &str,
    count: usize,
) -> anyhow::Result<Vec<AgentSlot>> {
    let roster = crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
    let roster_len = roster.len();
    let next_idx = i64::try_from(roster_len).unwrap_or(i64::MAX);
    let slots = build_agent_slots(
        &ticket.id,
        Role::Analyst,
        prompt,
        &[],
        count,
        next_idx,
        count,
    );
    insert_round_slots(job_id, &slots, crate::jobs::AgentKind::Analyst).await;
    Ok(slots)
}

/// Normalize a blocker string for dedup.
fn normalize_blocker(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Aggregate the base round's blockers: the UNION of `issues_detected` from
/// sub-threshold verdicts, deduplicated by normalized text.
fn aggregate_blockers(results: &[ParallelVerdict]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for r in results {
        if let ParallelVerdict::Verdict(v) = r
            && v.score < ANALYST_PASS_THRESHOLD
        {
            for issue in &v.issues_detected {
                if seen.insert(normalize_blocker(issue)) {
                    out.push(issue.clone());
                }
            }
        }
    }
    out
}

/// Apply the escalation agents' verdicts to the blocker list.
fn apply_blocker_verification(
    blockers: &[String],
    verdicts: &[&crate::BlockerVerificationVerdict],
) -> Vec<String> {
    let mut judged = vec![0usize; blockers.len()];
    let mut refuted = vec![0usize; blockers.len()];
    let mut sharpened: Vec<Option<String>> = vec![None; blockers.len()];
    for round in verdicts {
        for item in &round.verdicts {
            judged[item.index] += 1;
            match item.verdict {
                crate::BlockerDisposition::Refuted => refuted[item.index] += 1,
                crate::BlockerDisposition::Sharpened => {
                    if sharpened[item.index].is_none() {
                        sharpened[item.index].clone_from(&item.sharpened_text);
                    }
                }
                crate::BlockerDisposition::Confirmed => {}
            }
        }
    }
    let mut out = Vec::new();
    for (i, blocker) in blockers.iter().enumerate() {
        let all_refuted = judged[i] > 0 && refuted[i] == judged[i];
        if all_refuted {
            continue;
        }
        out.push(sharpened[i].clone().unwrap_or_else(|| blocker.clone()));
    }
    out
}

/// Number the blocker list for the escalation prompt template.
fn format_blocker_list(blockers: &[String]) -> String {
    blockers
        .iter()
        .enumerate()
        .map(|(i, b)| format!("{i}. {b}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render the escalation round's verification outcome.
fn format_blocker_verification_report(blockers: &[String], final_blockers: &[String]) -> String {
    let mut out = String::from("\n\n### Blocker verification\n");
    if final_blockers.is_empty() {
        out.push_str(
            "All flagged blockers were refuted by the verification round — no blockers remain.",
        );
    } else {
        let _ = writeln!(
            out,
            "{} of {} flagged blocker(s) survived verification. Remaining actionable blockers:",
            final_blockers.len(),
            blockers.len(),
        );
        for (i, b) in final_blockers.iter().enumerate() {
            let _ = write!(out, "\n{}. {b}", i + 1);
        }
    }
    out
}

/// Append the escalation round's verdict to the joint comment.
fn append_escalation_report(
    joint_comment: &mut String,
    base_results: &[ParallelVerdict],
    escalation_results: &[ParallelVerdict],
) {
    let verification: Vec<&crate::BlockerVerificationVerdict> = escalation_results
        .iter()
        .filter_map(|r| match r {
            ParallelVerdict::BlockerVerification(v) => Some(v),
            _ => None,
        })
        .collect();
    let dispatched = escalation_results.len();
    let succeeded = verification.len();
    if succeeded == 0 {
        if dispatched > 0 {
            joint_comment.push_str(
                "\n\n### Blocker verification\n\
                 The verification round could not produce a verdict — the base-round \
                 blockers remain unverified.",
            );
        }
        return;
    }
    let blockers = aggregate_blockers(base_results);
    let final_blockers = apply_blocker_verification(&blockers, &verification);
    joint_comment.push_str(&format_blocker_verification_report(
        &blockers,
        &final_blockers,
    ));
    if succeeded < dispatched {
        let _ = write!(
            joint_comment,
            "\n\nNote: only {succeeded} of {dispatched} verifier(s) returned a verdict; the \
             failed verifier(s) did not participate."
        );
    }
}

/// Format a natural-language summary of analyst verdict categories.
fn format_analyst_summary(
    total: usize,
    lgtm: usize,
    minor_issues: usize,
    potential_blockers: usize,
    missing_analysis: usize,
) -> String {
    let description = [
        (lgtm, "LGTM"),
        (minor_issues, "found minor issues"),
        (potential_blockers, "flagged potential blockers"),
        (missing_analysis, "provided no analysis"),
    ]
    .iter()
    .filter(|&&(count, _label)| count > 0)
    .map(|&(count, label)| {
        if count == total {
            format!("All {label}")
        } else {
            format!("{count} {label}")
        }
    })
    .collect::<Vec<_>>()
    .join(", ");

    format!("{total} analysts reviewed this ticket. {description}.")
}

/// Escalate a base analysis round that flagged blockers: run 2 blocker-
/// verification analysts on the SAME phase job, extending `results`. Returns
/// `false` when the ticket left the Analysis phase; sets `paused` when the
/// escalation round itself was pause-frozen.
#[must_use]
async fn maybe_escalate_analysis(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    _resumed: bool,
    results: &mut Vec<ParallelVerdict>,
    paused: &mut bool,
) -> bool {
    let blockers = aggregate_blockers(results);
    if blockers.is_empty() {
        return true;
    }
    if crate::shutdown::aborting() {
        return true;
    }
    if guard_job_phase(&ticket.id, TicketPhase::Analysis, job_id).await {
        return false;
    }
    info!(
        ticket = %ticket.id,
        blockers = blockers.len(),
        "Base analysis flagged blockers — escalating with 2 additional analysts",
    );
    let escalation_task = substitute(
        &load_prompt("analyze/blocker_verification.md"),
        &[("{{blockers}}", &format_blocker_list(&blockers))],
    );
    let extra_slots = match append_analysis_slots(ticket, job_id, &escalation_task, 2).await {
        Ok(s) => s,
        Err(e) => {
            warn!(ticket = %ticket.id, error = %e, "Failed to append escalation slots — proceeding with base round");
            Vec::new()
        }
    };
    if extra_slots.is_empty() {
        return true;
    }
    let extraction_prompt = load_prompt("extraction/blocker_verification.md");
    let blockers_arc = std::sync::Arc::<[String]>::from(blockers);
    let (extra, extra_paused) = run_parallel_agents(
        ticket,
        ws,
        Role::Analyst,
        &extraction_prompt,
        ExtractionMode::BlockerVerification {
            blockers: blockers_arc,
        },
        job_id,
        &extra_slots,
        false,
        TicketPhase::Analysis,
    )
    .await;
    results.extend(extra);
    *paused |= extra_paused;
    true
}

/// Spawn 3 parallel analyst agents (base) to research a backlog ticket on the
/// phase job, escalating to 5 when the base round aggregates blockers.
async fn dispatch_backlog_analysts(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    let prompt_key = if ticket.reporter == Role::Maintainer.as_str() {
        "analyze/maintainer_ticket.md"
    } else {
        "analyze/manager_ticket.md"
    };
    let message = load_prompt(prompt_key);

    let slots =
        ensure_analysis_slots(&ticket, job_id, &message, DEFAULT_PARALLEL_AGENT_COUNT).await;
    run_analysis_round(&ticket, &ws, job_id, &slots, false).await;
}

/// Run an analysis round: parallel analysts, optional blocker-verification
/// escalation, then verdict finalization.
async fn run_analysis_round(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    slots: &[AgentSlot],
    resumed: bool,
) {
    let extraction_prompt = load_prompt("extraction/analyst.md");
    let (base_results, base_paused) = run_parallel_agents(
        ticket,
        ws,
        Role::Analyst,
        &extraction_prompt,
        ExtractionMode::ScoreVerdict,
        job_id,
        slots,
        resumed,
        TicketPhase::Analysis,
    )
    .await;
    if base_paused {
        pause_freezing(ticket, job_id).await;
        return;
    }
    let base_count = base_results.len();
    let mut results = base_results;
    let mut paused = false;
    if !maybe_escalate_analysis(ticket, ws, job_id, resumed, &mut results, &mut paused).await {
        return;
    }
    if paused {
        pause_freezing(ticket, job_id).await;
        return;
    }
    let (base_results, escalation_results) = results.split_at(base_count);
    finalize_analysis_round(ws, ticket, base_results, escalation_results, job_id).await;
}

/// Finalize an analysis round — always advances to Planning (fail-open).
async fn finalize_analysis_round(
    ws: &Workspace,
    ticket: &Ticket,
    base_results: &[ParallelVerdict],
    escalation_results: &[ParallelVerdict],
    job_id: &str,
) {
    if guard_job_phase(&ticket.id, TicketPhase::Analysis, job_id).await {
        return;
    }
    if crate::shutdown::aborting() {
        return;
    }
    process_analyst_verdicts(ws, ticket, base_results, escalation_results, job_id).await;
    if crate::shutdown::aborting() {
        return;
    }
    if let Err(e) = crate::jobs::complete_ticket_job(&crate::session::store().conn, job_id).await {
        warn!(job = %job_id, error = %e, "Failed to terminalize analysis job");
    }
}

/// Hard technical failure: no analyst produced usable output. Reset the
/// attempt — the ticket stays in Analysis for a fresh round. No workspace
/// pause (Analysis is not an implementation phase), no retry limit, no bounce
/// budget.
async fn reset_analysis_round(ticket: &Ticket, job_id: &str) {
    info!(
        ticket = %ticket.id,
        "Backlog analysis produced no usable output — resetting for a fresh attempt",
    );
    let comment = "Backlog analysis produced no usable output — resetting for a fresh attempt.";
    reset_phase_attempt(
        ticket,
        TicketPhase::Analysis,
        job_id,
        "analysis failure",
        comment,
    )
    .await;
}

/// Process the analyst verdicts and transition the ticket to Planning
/// (fail-open — the ticket always advances, the Manager decides). The one
/// hard-failure exception: when NO analyst produced usable output
/// (`extracted_count == 0`), reset the attempt (stay in Analysis, fresh
/// round, no workspace pause) instead of advancing with nothing.
async fn process_analyst_verdicts(
    ws: &Workspace,
    ticket: &Ticket,
    base_results: &[ParallelVerdict],
    escalation_results: &[ParallelVerdict],
    job_id: &str,
) {
    let nonempty_count = base_results
        .iter()
        .filter(|r| !matches!(r, ParallelVerdict::NoResponse(_)))
        .count();
    let dispatched = base_results.len();
    let mut lgtm = 0usize;
    let mut minor_issues = 0usize;
    let mut potential_blockers = 0usize;
    let mut missing_analysis = 0usize;

    for r in base_results {
        match r {
            ParallelVerdict::Verdict(v)
                if v.score >= ANALYST_PASS_THRESHOLD && v.issues_detected.is_empty() =>
            {
                lgtm += 1;
            }
            ParallelVerdict::Verdict(v) if v.score >= ANALYST_PASS_THRESHOLD => minor_issues += 1,
            ParallelVerdict::Verdict(_) => potential_blockers += 1,
            ParallelVerdict::NoResponse(_) | ParallelVerdict::ParseFailed(_) => {
                missing_analysis += 1;
            }
            ParallelVerdict::BlockerVerification(_) => unreachable!(),
        }
    }

    let extracted_count = dispatched - missing_analysis;

    if crate::shutdown::aborting() {
        return;
    }

    if extracted_count == 0 {
        reset_analysis_round(ticket, job_id).await;
        return;
    }

    let summary = format_analyst_summary(
        dispatched,
        lgtm,
        minor_issues,
        potential_blockers,
        missing_analysis,
    );
    let passing_count = lgtm + minor_issues;
    let all_passed = passing_count == dispatched;

    let mut joint_comment = build_round_joint_comment(
        stage_name(Role::Analyst),
        base_results,
        ANALYST_PASS_THRESHOLD,
        Role::Analyst,
        &summary,
        ws,
        &ticket.id,
        &ticket.title,
    )
    .await;

    append_escalation_report(&mut joint_comment, base_results, escalation_results);

    if !matches!(
        with_comment_and_transition(
            TransitionCtx::notifying(
                ticket,
                TicketPhase::Analysis,
                TicketPhase::Planning,
                "Analyst"
            ),
            async |tx| {
                BoardStore::add_comment_tx(
                    tx,
                    &ticket.id,
                    stage_name(Role::Analyst),
                    &joint_comment,
                )
                .await?;
                Ok(())
            },
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        return;
    }

    if all_passed {
        info!(
            ticket = %ticket.id,
            nonempty_count,
            "Backlog analysis complete — all analysts passed (≥ {ANALYST_PASS_THRESHOLD}/10)",
        );
    } else {
        info!(
            ticket = %ticket.id,
            nonempty_count,
            extracted_count,
            passing_count,
            "Backlog analysis incomplete — moved to planning ({nonempty_count}/{dispatched} responded, \
             {extracted_count} extracted, {passing_count} passed)",
        );
    }
}
