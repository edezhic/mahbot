//! Analysis phase module — parallel analysts research a backlog ticket.

use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::{Role, Workspace};

use super::{
    AgentSlot, BoardStore, ExtractionMode, FinalizeOutcome, ParallelVerdict, TicketPhase,
    TransitionCtx, Write, agent_slot_from_roster_row, build_agent_slots, build_round_joint_comment,
    deserialize_verdict_outcome, guard_job_phase, info, insert_round_slots, pause_freezing,
    reset_phase_attempt, run_parallel_agents, stage_name, warn, with_comment_and_transition,
};

/// Default number of parallel analyst agents per round. Reviewers use a
/// calibrated dynamic count (see [`crate::pipeline::verdict::review_agent_count`]);
/// QA runs a single tester (see [`qa::QA_PARALLEL_AGENT_COUNT`]).
const DEFAULT_PARALLEL_AGENT_COUNT: usize = 3;

/// Minimum acceptable verification score (0-10) for analyst verdicts.
const ANALYST_PASS_THRESHOLD: u8 = 7;

/// Load the three base ticket-analysis angle sections for Backlog analysts.
fn load_ticket_analysis_angles() -> Vec<String> {
    load_prompt_sections("analyze/ticket_angles.md")
}

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, TicketPhase::Analysis, &job_id).await {
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
        0,
        count,
    );
    insert_round_slots(job_id, &slots, crate::jobs::AgentKind::Analyst).await;
    slots
}

/// Append escalation slots (3, 4) to an existing phase job. Escalation slots
/// are purely blocker-verification focused and inherit NO base angle. The
/// starting idx derives from the max stored idx (not the roster length) so a
/// resumed roster with prior rows never collides or over-indexes.
async fn append_analysis_slots(
    ticket: &Ticket,
    job_id: &str,
    prompt: &str,
    count: usize,
) -> anyhow::Result<Vec<AgentSlot>> {
    let roster = crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
    let next_idx = roster
        .iter()
        .filter_map(|r| r.idx)
        .max()
        .map_or(0, |m| m + 1);
    let slots = build_agent_slots(&ticket.id, Role::Analyst, prompt, &[], next_idx, count);
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

/// A merged substance outcome for one aggregate blocker after the escalation
/// round. The two verifiers are reduced deterministically and conservatively
/// into a single grade; every aggregated blocker survives (enrichment-only).
pub(crate) struct ResolvedBlocker {
    pub text: String,
    pub kind: crate::BlockerKind,
    pub severity: crate::BlockerSeverity,
    pub impact: String,
    pub reasoning: String,
}

/// Human label for a blocker kind (display only; never stored).
fn display_kind(kind: crate::BlockerKind) -> &'static str {
    match kind {
        crate::BlockerKind::MainPathBlocker => "main-path blocker",
        crate::BlockerKind::RiskEdgeCase => "risk/edge-case",
    }
}

/// Human label for a blocker severity (display only; never stored).
fn display_severity(severity: crate::BlockerSeverity) -> &'static str {
    match severity {
        crate::BlockerSeverity::Low => "low",
        crate::BlockerSeverity::Medium => "medium",
        crate::BlockerSeverity::High => "high",
        crate::BlockerSeverity::Critical => "critical",
    }
}

/// Merge the escalation verifiers' substance verdicts into one per-blocker
/// outcome. Deterministic and conservative: severity is the max across
/// verifiers, kind is `main_path_blocker` if any verifier chose it, and
/// impact/reasoning concatenate the non-empty per-verifier values separated by
/// " — ". A verifier that produced no verdict for a blocker contributes
/// nothing. Every aggregated blocker survives with an enriched grade (this is
/// never a filter).
pub(crate) fn apply_blocker_verification(
    blockers: &[String],
    verdicts: &[&crate::BlockerVerificationVerdict],
) -> Vec<ResolvedBlocker> {
    blockers
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut severity = crate::BlockerSeverity::Low;
            let mut kind = crate::BlockerKind::RiskEdgeCase;
            let mut impact_parts: Vec<&str> = Vec::new();
            let mut reasoning_parts: Vec<&str> = Vec::new();
            for round in verdicts {
                for item in &round.verdicts {
                    if item.index != i {
                        continue;
                    }
                    if item.severity > severity {
                        severity = item.severity;
                    }
                    if item.kind == crate::BlockerKind::MainPathBlocker {
                        kind = item.kind;
                    }
                    if !item.impact.trim().is_empty() {
                        impact_parts.push(item.impact.trim());
                    }
                    if !item.reasoning.trim().is_empty() {
                        reasoning_parts.push(item.reasoning.trim());
                    }
                }
            }
            ResolvedBlocker {
                text: text.clone(),
                kind,
                severity,
                impact: impact_parts.join(" — "),
                reasoning: reasoning_parts.join(" — "),
            }
        })
        .collect()
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

/// Render the escalation round's substance outcome. Always reached with a
/// non-empty blocker list (escalation only appends when blockers aggregate).
fn format_blocker_verification_report(blockers: &[ResolvedBlocker]) -> String {
    let mut out = String::from("\n\n### Blocker verification\n");
    let _ = writeln!(
        out,
        "{} flagged blocker(s) verified with substance grades:",
        blockers.len()
    );
    for (i, b) in blockers.iter().enumerate() {
        let _ = writeln!(
            out,
            "\n{}. {} — {} (severity: {})",
            i + 1,
            b.text,
            display_kind(b.kind),
            display_severity(b.severity),
        );
        let _ = write!(out, "\n   Impact: {}", b.impact);
        let _ = write!(out, "\n   Reasoning: {}", b.reasoning);
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
    let resolved = apply_blocker_verification(&blockers, &verification);
    joint_comment.push_str(&format_blocker_verification_report(&resolved));
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
///
/// This is the FRESH-append path only. An interrupted escalation round is
/// resumed by [`dispatch_backlog_analysts`] via [`resume_escalation_round`],
/// which reconstructs the frozen base verdicts and re-runs only the not-Done
/// escalation slots from the stored roster — so no duplicate-idx rows
/// accumulate here and no completed escalation work is re-dispatched.
#[must_use]
async fn maybe_escalate_analysis(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
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
            return true;
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
        TicketPhase::Analysis,
        false,
    )
    .await;
    results.extend(extra);
    *paused |= extra_paused;
    true
}

/// Spawn 3 parallel analyst agents (base) to research a backlog ticket on the
/// phase job, escalating to 5 when the base round aggregates blockers.
///
/// On a re-drive of an interrupted round (non-empty roster) this slot-resumes:
/// an incomplete base round re-runs its not-Done base slots from their stored
/// tasks; a base round that already appended escalation slots has its base
/// verdicts reconstructed from stored outcomes (freezing the blocker list) and
/// only the not-Done escalation slots are re-run.
async fn dispatch_backlog_analysts(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    let prompt_key = if ticket.reporter == Role::Maintainer.as_str() {
        "analyze/maintainer_ticket.md"
    } else {
        "analyze/manager_ticket.md"
    };
    let message = load_prompt(prompt_key);

    let conn = &crate::session::store().conn;
    let roster = match crate::jobs::list_agents_for_job(conn, job_id).await {
        Ok(roster) => roster,
        Err(e) => {
            // A read failure must NOT degrade to a fresh dispatch: doing so would
            // re-derive a new-suffix roster and orphan the interrupted round's
            // rows. Bail and let the poller re-drive (the job stays occupied,
            // with no running agents, so re-dispatch is safe).
            warn!(ticket = %ticket.id, error = %e, "Failed to read analysis roster — bailing to preserve interrupted round");
            return;
        }
    };
    if roster.is_empty() {
        let slots =
            ensure_analysis_slots(&ticket, job_id, &message, DEFAULT_PARALLEL_AGENT_COUNT).await;
        run_analysis_round(&ticket, &ws, job_id, &slots, false).await;
        return;
    }

    // Resume: an interrupted round lives on in the roster. Split the base cohort
    // (idx < the base count) from any escalation cohort (idx >= the base count).
    let base_count = i64::try_from(DEFAULT_PARALLEL_AGENT_COUNT).unwrap_or(i64::MAX);
    let escalation_rows: Vec<&crate::jobs::AgentRow> = roster
        .iter()
        .filter(|r| r.idx.unwrap_or(0) >= base_count)
        .collect();
    if escalation_rows.is_empty() {
        // Base round not yet escalated: re-run the not-Done base slots with their
        // stored tasks, then re-evaluate escalation on the resumed base verdicts.
        // Re-arm the not-Done slots as launched so comment routing and the
        // Running Agents view track the in-flight members while the round runs.
        let base_slots: Vec<AgentSlot> = roster.iter().map(agent_slot_from_roster_row).collect();
        let not_done: Vec<String> = base_slots
            .iter()
            .filter(|s| s.status != crate::jobs::RowStatus::Done)
            .map(|s| s.agent_id.clone())
            .collect();
        if let Err(e) = crate::jobs::rearm_roster_launched(conn, job_id, &not_done).await {
            warn!(ticket = %ticket.id, error = %e, "Failed to re-arm resumed analysis base slots");
        }
        run_analysis_round(&ticket, &ws, job_id, &base_slots, true).await;
    } else {
        // Base round completed (it triggered the escalation). Reconstruct the
        // base verdicts from stored outcomes — NOT a re-run — so the blocker list
        // that the escalation slots were built for stays frozen; only the not-Done
        // escalation slots resume.
        let base_results: Vec<ParallelVerdict> = roster
            .iter()
            .filter(|r| r.idx.unwrap_or(0) < base_count)
            .map(|r| deserialize_verdict_outcome(r.outcome.as_deref().unwrap_or("")))
            .collect();
        resume_escalation_round(&ticket, &ws, job_id, base_results).await;
    }
}

/// Resume an escalation round whose base verdicts are already frozen. The
/// already-Done escalation slots are reconstructed from stored outcomes and the
/// not-Done ones re-run with their stored tasks; the blocker list is re-derived
/// from the frozen base verdicts (it is never stored in the roster).
async fn resume_escalation_round(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    base_results: Vec<ParallelVerdict>,
) {
    let blockers = aggregate_blockers(&base_results);
    if blockers.is_empty() {
        // No blockers reconstructable — the escalation slots are orphaned by the
        // interruption. Finalize fail-open on the base verdicts alone.
        let (base, escalation) = (base_results.as_slice(), &[][..]);
        finalize_analysis_round(ws, ticket, base, escalation, job_id).await;
        return;
    }
    if crate::shutdown::aborting()
        || guard_job_phase(&ticket.id, TicketPhase::Analysis, job_id).await
    {
        return;
    }
    let conn = &crate::session::store().conn;
    let roster = match crate::jobs::list_agents_for_job(conn, job_id).await {
        Ok(roster) => roster,
        Err(e) => {
            warn!(ticket = %ticket.id, error = %e, "Failed to read escalation roster — retreating to base-only finalize");
            let (base, escalation) = (base_results.as_slice(), &[][..]);
            finalize_analysis_round(ws, ticket, base, escalation, job_id).await;
            return;
        }
    };
    let base_count = i64::try_from(DEFAULT_PARALLEL_AGENT_COUNT).unwrap_or(i64::MAX);
    let escalation_slots: Vec<AgentSlot> = roster
        .iter()
        .filter(|r| r.idx.unwrap_or(0) >= base_count)
        .map(agent_slot_from_roster_row)
        .collect();
    if escalation_slots.is_empty() {
        // The roster's escalation cohort vanished (job raced through a reset) —
        // finalize on the base verdicts alone.
        let (base, escalation) = (base_results.as_slice(), &[][..]);
        finalize_analysis_round(ws, ticket, base, escalation, job_id).await;
        return;
    }
    let not_done: Vec<String> = escalation_slots
        .iter()
        .filter(|s| s.status != crate::jobs::RowStatus::Done)
        .map(|s| s.agent_id.clone())
        .collect();
    if let Err(e) = crate::jobs::rearm_roster_launched(conn, job_id, &not_done).await {
        warn!(ticket = %ticket.id, error = %e, "Failed to re-arm resumed escalation roster slots");
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
        &escalation_slots,
        TicketPhase::Analysis,
        true,
    )
    .await;
    if extra_paused {
        pause_freezing(ticket, job_id).await;
        return;
    }
    finalize_analysis_round(ws, ticket, &base_results, &extra, job_id).await;
}

/// Run an analysis round: parallel analysts, optional blocker-verification
/// escalation, then verdict finalization.
///
/// `resume` marks a slot-resume round: the not-Done base slots continue their
/// sessions and the leader-stagger is skipped.
async fn run_analysis_round(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    slots: &[AgentSlot],
    resume: bool,
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
        TicketPhase::Analysis,
        resume,
    )
    .await;
    if base_paused {
        pause_freezing(ticket, job_id).await;
        return;
    }
    let base_count = base_results.len();
    let mut results = base_results;
    let mut paused = false;
    if !maybe_escalate_analysis(ticket, ws, job_id, &mut results, &mut paused).await {
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
    let comment = "Backlog analysis produced no usable output.";
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
