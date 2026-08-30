//! Analysis phase module — parallel analysts research a backlog ticket.

use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::{Role, Workspace};

use super::{
    AgentSlot, BoardStore, ExtractionMode, FinalizeOutcome, JointRound, ParallelVerdict,
    TicketPhase, TransitionCtx, Write, agent_slot_from_roster_row, build_agent_slots,
    build_round_grouping, deserialize_verdict_outcome, guard_job_phase, info, insert_round_slots,
    issue_grade, pause_freezing, render_joint_comment, reset_phase_attempt, run_parallel_agents,
    stage_name, warn, with_comment_and_transition,
};

/// Default number of parallel analyst agents per round. Reviewers use a
/// calibrated dynamic count (see [`crate::pipeline::verdict::review_agent_count`]);
/// QA runs a single tester (see [`qa::QA_PARALLEL_AGENT_COUNT`]).
const DEFAULT_PARALLEL_AGENT_COUNT: usize = 3;

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

/// One escalated deduplicated group selected for the blocker-verification
/// round. `text` is the display string baked into the verifier instructions:
/// "{heading}: {representative}" for a grouped entry, or just the issue text
/// for a lone (ungrouped / fallback) blocker. The report renders `text`
/// directly, so no prose-splitting is needed to recover structure on resume.
pub(crate) struct EscalationEntry {
    pub text: String,
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
    entries: &[EscalationEntry],
    verdicts: &[&crate::BlockerVerificationVerdict],
) -> Vec<ResolvedBlocker> {
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
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
                text: entry.text.clone(),
                kind,
                severity,
                impact: impact_parts.join(" — "),
                reasoning: reasoning_parts.join(" — "),
            }
        })
        .collect()
}

/// Marker emitted by `format_blocker_list` before the numbered entries. The
/// resume parser anchors on THIS line, not on the `analyze/blocker_verification.md`
/// prompt's lead-in prose — co-located with the generator so edits to that
/// prompt asset cannot silently break crash-resume recovery.
const BLOCKER_LIST_MARKER: &str = "Blocker list:";

/// Legacy prompt lead-in phrase (`analyze/blocker_verification.md`) that
/// preceded the `{{blockers}}` list in tasks baked before [`BLOCKER_LIST_MARKER`]
/// existed. Kept as a resume fallback so an escalation task stored by the
/// previous code version still recovers on a resume-across-deploy.
const LEGACY_BLOCKER_LIST_MARKER: &str = "matching index.";

/// Number the blocker-verification entries for the escalation prompt template.
fn format_blocker_list(entries: &[EscalationEntry]) -> String {
    let mut out = format!("{BLOCKER_LIST_MARKER}\n");
    for (i, e) in entries.iter().enumerate() {
        let text = e.text.replace('\r', "").replace('\n', " ");
        let _ = writeln!(out, "{i}. {text}");
    }
    out
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
    entries: &[EscalationEntry],
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
    let resolved = apply_blocker_verification(entries, &verification);
    joint_comment.push_str(&format_blocker_verification_report(&resolved));
    if succeeded < dispatched {
        let _ = write!(
            joint_comment,
            "\n\nNote: only {succeeded} of {dispatched} verifier(s) returned a verdict; the \
             failed verifier(s) did not participate."
        );
    }
}

/// Format a natural-language grade-count summary of analyst verdicts.
fn format_grade_summary(results: &[ParallelVerdict]) -> String {
    let mut clean = 0usize;
    let mut minor = 0usize;
    let mut major = 0usize;
    let mut blocker = 0usize;
    let mut missing = 0usize;
    for r in results {
        match r {
            ParallelVerdict::Analysis(v) => {
                if v.issues_detected.is_empty() {
                    clean += 1;
                } else {
                    match v.issues_detected.iter().map(|a| a.grade).max() {
                        Some(crate::IssueGrade::Blocker) => blocker += 1,
                        Some(crate::IssueGrade::Major) => major += 1,
                        _ => minor += 1,
                    }
                }
            }
            ParallelVerdict::NoResponse(_) | ParallelVerdict::ParseFailed(_) => missing += 1,
            _ => {}
        }
    }
    let total = results.len();
    let description = [
        (blocker, "flagged a blocker"),
        (major, "found major issues"),
        (minor, "found minor issues"),
        (clean, "found no issues"),
        (missing, "provided no analysis"),
    ]
    .iter()
    .filter(|&&(count, _)| count > 0)
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

/// Build the escalation entries (whole deduplicated groups) from the round's
/// grouping outcome. A group escalates when ANY member (including its
/// collapsed same-fact ids) carries `grade = blocker`; the entry uses the
/// group's heading with the FIRST blocker-bearing member's text as its
/// display. Ungrouped blockers fail open and escalate as lone entries. A
/// `Fallback` outcome (no grouping) escalates every blocker item as a lone
/// entry — the old deterministic escalation never depended on grouping.
fn escalation_groups(
    round: &JointRound,
    outcome: &crate::consensus::RepairOutcome,
) -> Vec<EscalationEntry> {
    let table = crate::consensus::ItemTable::new(&round.issues);
    // First blocker-bearing id within a set of members (incl. collapsed ids),
    // scanning representative then collapsed ids. `None` when none is blocker.
    let first_blocker = |members: &[crate::consensus::GroupingMember]| {
        for member in members {
            for id in member.ids() {
                if issue_grade(round, &table, id) == Some(crate::IssueGrade::Blocker) {
                    return Some(id);
                }
            }
        }
        None
    };
    match outcome {
        crate::consensus::RepairOutcome::Repaired { output, .. } => {
            // Whole deduplicated groups escalate; the entry uses the group
            // heading with the first blocker member's text as its display.
            let mut entries = Vec::new();
            for group in &output.groups {
                if let Some(id) = first_blocker(&group.members) {
                    let member_text = table
                        .resolve(id)
                        .map(|(_, t)| t.to_string())
                        .unwrap_or_default();
                    entries.push(EscalationEntry {
                        text: format!("{}: {}", group.heading, member_text),
                    });
                }
            }
            // Ungrouped blockers still escalate (fail-open): the LLM left a
            // genuine blocker thematically isolated, but the most severe
            // finding must still be substance-verified.
            for member in &output.ungrouped {
                if first_blocker(std::slice::from_ref(member)).is_some()
                    && let Some((_, text)) = table.resolve(member.id)
                {
                    entries.push(EscalationEntry {
                        text: text.to_string(),
                    });
                }
            }
            entries
        }
        crate::consensus::RepairOutcome::Fallback => {
            // No LLM grouping (exhausted or clean): escalate every blocker
            // item as a lone entry — the old deterministic escalation never
            // depended on grouping success.
            let mut entries = Vec::new();
            for id in 0..table.len() {
                if issue_grade(round, &table, id) == Some(crate::IssueGrade::Blocker)
                    && let Some((_, text)) = table.resolve(id)
                {
                    entries.push(EscalationEntry {
                        text: text.to_string(),
                    });
                }
            }
            entries
        }
    }
}

/// Parse the numbered blocker list baked into a stored escalation slot's task.
/// The list is anchored on [`BLOCKER_LIST_MARKER`] (co-located with the
/// generator so prompt-asset edits cannot silently break crash-resume
/// recovery), falling back to the legacy `matching index.` lead-in for tasks
/// baked by a previous code version; each following trimmed line matching
/// `^\d+\.\s+(.+)$` yields one entry whose text is the full displayed line.
/// Returns in order.
fn parse_escalation_entries(task: &str) -> Vec<EscalationEntry> {
    let anchor = task
        .find(BLOCKER_LIST_MARKER)
        .map(|p| (p, BLOCKER_LIST_MARKER.len()))
        .or_else(|| {
            task.find(LEGACY_BLOCKER_LIST_MARKER)
                .map(|p| (p, LEGACY_BLOCKER_LIST_MARKER.len()))
        });
    let Some((start, marker_len)) = anchor else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for line in task[start + marker_len..].lines() {
        let Some(text) = split_numbered(line.trim()) else {
            continue;
        };
        entries.push(EscalationEntry {
            text: text.to_string(),
        });
    }
    entries
}

/// Split a trimmed line matching `^\d+\.\s+(.+)$` into the captured text after
/// the leading run of digits, the `.`, and the mandatory whitespace.
fn split_numbered(line: &str) -> Option<&str> {
    let after_digits = line.trim_start_matches(|c: char| c.is_ascii_digit());
    if after_digits.len() == line.len() {
        return None;
    }
    let after_dot = after_digits.strip_prefix('.')?;
    let after_ws = after_dot.trim_start();
    if after_ws.is_empty() {
        return None;
    }
    Some(after_ws)
}

/// Escalate a base analysis round that flagged blocker groups: run 2 blocker-
/// verification analysts on the SAME phase job, extending `results`. Returns
/// `false` when the ticket left the Analysis phase; sets `paused` when the
/// escalation round itself was pause-frozen.
///
/// This is the FRESH-append path only. An interrupted escalation round is
/// resumed by [`dispatch_backlog_analysts`] via [`resume_escalation_round`],
/// which reconstructs the frozen base verdicts and re-runs only the not-Done
/// escalation slots from the stored roster — so no duplicate-idx rows
/// accumulate here and no completed escalation work is re-dispatched.
#[expect(clippy::too_many_arguments)]
#[must_use]
async fn maybe_escalate_analysis(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    results: &mut Vec<ParallelVerdict>,
    paused: &mut bool,
    round: &JointRound,
    outcome: &crate::consensus::RepairOutcome,
    escalation_entries: &mut Vec<EscalationEntry>,
) -> bool {
    let entries = escalation_groups(round, outcome);
    if entries.is_empty() {
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
        blockers = entries.len(),
        "Base analysis flagged blocker groups — escalating with 2 additional analysts",
    );
    let escalation_task = substitute(
        &load_prompt("analyze/blocker_verification.md"),
        &[("{{blockers}}", &format_blocker_list(&entries))],
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
    let blockers_arc = std::sync::Arc::<[String]>::from(
        entries.iter().map(|e| e.text.clone()).collect::<Vec<_>>(),
    );
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
    *escalation_entries = entries;
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

/// Build the base-round grouping and finalize: the resume fallthrough arms
/// (roster-bail, vanished escalation cohort, orphaned escalation slots) pass an
/// empty escalation report, while the normal resume path passes the re-run
/// base grouping and the baked-in escalation entries.
async fn finalize_analysis_round_with_grouping(
    ws: &Workspace,
    ticket: &Ticket,
    base_results: &[ParallelVerdict],
    escalation_results: &[ParallelVerdict],
    escalation_entries: &[EscalationEntry],
    job_id: &str,
) {
    let summary = format_grade_summary(base_results);
    let (round, outcome) = build_round_grouping(
        "Analysis",
        base_results,
        /* threshold unused for analysis */ 0,
        Role::Analyst,
        &summary,
        ws,
        &ticket.id,
        &ticket.title,
    )
    .await;
    finalize_analysis_round(
        ticket,
        base_results,
        escalation_results,
        &round,
        &outcome,
        escalation_entries,
        job_id,
    )
    .await;
}

/// Resume an escalation round whose base verdicts are already frozen. The
/// already-Done escalation slots are reconstructed from stored outcomes and the
/// not-Done ones re-run with their stored tasks; the escalation group list is
/// re-derived from the task baked into the first escalation slot (it is never
/// re-run through consolidation).
async fn resume_escalation_round(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    base_results: Vec<ParallelVerdict>,
) {
    let conn = &crate::session::store().conn;
    let roster = match crate::jobs::list_agents_for_job(conn, job_id).await {
        Ok(roster) => roster,
        Err(e) => {
            // A read failure cannot degrade to a fresh dispatch: the escalation
            // entries are baked into the stored tasks, and a re-derivation would
            // orphan the interrupted round's rows. Retreat to base-only finalize.
            warn!(ticket = %ticket.id, error = %e, "Failed to read escalation roster — retreating to base-only finalize");
            finalize_analysis_round_with_grouping(ws, ticket, &base_results, &[], &[], job_id)
                .await;
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
        finalize_analysis_round_with_grouping(ws, ticket, &base_results, &[], &[], job_id).await;
        return;
    }
    let entries = parse_escalation_entries(&escalation_slots[0].task);
    if entries.is_empty() {
        // Orphaned escalation slots (no blocker list baked into their task) —
        // the base round still advances fail-open.
        finalize_analysis_round_with_grouping(ws, ticket, &base_results, &[], &[], job_id).await;
        return;
    }
    if crate::shutdown::aborting()
        || guard_job_phase(&ticket.id, TicketPhase::Analysis, job_id).await
    {
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
    let blockers_arc = std::sync::Arc::<[String]>::from(
        entries.iter().map(|e| e.text.clone()).collect::<Vec<_>>(),
    );
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
    // Re-run the base consolidation for the comment body; the escalation report
    // uses the baked-in entries (not a re-run consolidation) — decision D.
    finalize_analysis_round_with_grouping(ws, ticket, &base_results, &extra, &entries, job_id)
        .await;
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
        ExtractionMode::ScorelessVerdict,
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
    let summary = format_grade_summary(&base_results);
    let (round, outcome) = build_round_grouping(
        "Analysis",
        &base_results,
        /* threshold unused for analysis */ 0,
        Role::Analyst,
        &summary,
        ws,
        &ticket.id,
        &ticket.title,
    )
    .await;
    let mut results = base_results;
    let mut escalation_entries: Vec<EscalationEntry> = Vec::new();
    let mut paused = false;
    if !maybe_escalate_analysis(
        ticket,
        ws,
        job_id,
        &mut results,
        &mut paused,
        &round,
        &outcome,
        &mut escalation_entries,
    )
    .await
    {
        return;
    }
    if paused {
        pause_freezing(ticket, job_id).await;
        return;
    }
    let (base_results, escalation_results) = results.split_at(base_count);
    finalize_analysis_round(
        ticket,
        base_results,
        escalation_results,
        &round,
        &outcome,
        &escalation_entries,
        job_id,
    )
    .await;
}

/// Finalize an analysis round — always advances to Planning (fail-open).
async fn finalize_analysis_round(
    ticket: &Ticket,
    base_results: &[ParallelVerdict],
    escalation_results: &[ParallelVerdict],
    round: &JointRound,
    outcome: &crate::consensus::RepairOutcome,
    escalation_entries: &[EscalationEntry],
    job_id: &str,
) {
    if guard_job_phase(&ticket.id, TicketPhase::Analysis, job_id).await {
        return;
    }
    if crate::shutdown::aborting() {
        return;
    }
    process_analyst_verdicts(
        ticket,
        base_results,
        escalation_results,
        round,
        outcome,
        escalation_entries,
        job_id,
    )
    .await;
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
    ticket: &Ticket,
    base_results: &[ParallelVerdict],
    escalation_results: &[ParallelVerdict],
    round: &JointRound,
    outcome: &crate::consensus::RepairOutcome,
    escalation_entries: &[EscalationEntry],
    job_id: &str,
) {
    let missing_analysis = base_results
        .iter()
        .filter(|r| {
            matches!(
                r,
                ParallelVerdict::NoResponse(_) | ParallelVerdict::ParseFailed(_)
            )
        })
        .count();
    let dispatched = base_results.len();
    let extracted_count = dispatched - missing_analysis;
    if crate::shutdown::aborting() {
        return;
    }
    if extracted_count == 0 {
        reset_analysis_round(ticket, job_id).await;
        return;
    }
    let mut joint_comment = render_joint_comment(
        round,
        outcome,
        &crate::consensus::ItemTable::new(&round.issues),
    );
    append_escalation_report(&mut joint_comment, escalation_entries, escalation_results);
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
    info!(
        ticket = %ticket.id,
        nonempty_count = extracted_count,
        "Backlog analysis complete — moved to planning ({extracted_count}/{dispatched} extracted)",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the [`parse_escalation_entries`] dual-anchor contract: the stable
    /// code-owned [`BLOCKER_LIST_MARKER`] (baked by the current
    /// [`format_blocker_list`]) AND the legacy `matching index.` lead-in (tasks
    /// baked by a previous code version) both recover the full per-entry text,
    /// and a task with neither marker degrades to an empty list (fail-open).
    #[test]
    fn parse_escalation_entries_recover_marker_and_legacy_anchor() {
        let current = "report the outcome for each blocker using the matching index.\n\n\
                       Blocker list:\n\
                       0. Score-less verdict model: the shared type is unspecified\n\
                       1. Crash-resume reconciliation: the divergence is inconsistent";
        let entries = parse_escalation_entries(current);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].text,
            "Score-less verdict model: the shared type is unspecified"
        );
        assert_eq!(
            entries[1].text,
            "Crash-resume reconciliation: the divergence is inconsistent"
        );

        let legacy = "report the outcome for each blocker using the matching index.\n\n\
                      0. Score-less verdict model: the shared type is unspecified\n\
                      1. Crash-resume reconciliation: the divergence is inconsistent";
        let entries = parse_escalation_entries(legacy);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].text,
            "Score-less verdict model: the shared type is unspecified"
        );

        assert!(parse_escalation_entries("no blocker list anywhere").is_empty());
    }
}
