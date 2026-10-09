//! The `verification` stage runs the workspace's project commands, the read-only
//! code reviewers and the single functional tester at the same time on one phase
//! job, and the round produces ONE consolidated comment and one advance/bounce
//! decision. The project commands are nobody's roster slot: they are the round's
//! own work, run in parallel with the participants.
//!
//! The round's cohorts are told apart by their stored `agents.kind` labels, never
//! by slot order, and a roster that is not the round's shape is never resumed as
//! one. The identical-content skip, the churn-calibrated reviewer count and the
//! reviewed-base recording are reviewer-only: the tester is never skipped and
//! never records a reviewed base.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use crate::agent::role::DIAGNOSTICS_ROLE;
use crate::git::commands::{
    has_unstaged_changes, run_git_add_all, run_git_head, run_git_status, run_git_worktree_snapshot,
    run_git_write_tree,
};
use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::tools::shell::{ShellMode, ShellTool};
use crate::{DiagnosticsCommands, Role, Workspace};

use super::{
    AgentSlot, ExtractionMode, FinalizeOutcome, ParallelVerdict, TicketPhase, TransitionCtx,
    agent_slot_from_roster_row, bounce_to_development, build_agent_slots, build_round_grouping,
    comment_and_transition, debug, guard_job_phase, info, insert_round_slots, pause_freezing,
    read_roster_or_bail, render_joint_comment, reset_phase_attempt, run_parallel_agents,
    sync_phase_job_task, warn,
};

/// The stage's own name: the role the consolidated round comment is authored
/// under, the label in log fields and stage comments. (The recorded transition
/// actor is the phase token — `ACTIVE_PHASE.as_ref()`.)
pub(crate) const STAGE: &str = "Verification";

/// The phase a verification round works in.
const ACTIVE_PHASE: TicketPhase = TicketPhase::Verification;

/// The phase a clean round advances the ticket to.
const SUCCESS_PHASE: TicketPhase = TicketPhase::InSanitation;

/// Minimum acceptable score (0-10) for a verification verdict.
const VERIFICATION_THRESHOLD: u8 = 9;

/// The functional tester runs exactly one agent per round — the round's
/// functional check, which no round may run without.
const TESTER_COUNT: usize = 1;

/// Rendered at the head of the consolidated comment of a round whose reviewer
/// cohort was not dispatched: the skipped code review is part of the round's
/// result, so the reader must see it rather than infer it from its absence. It
/// ends with the blank line that separates it from the round's own sections.
const SKIPPED_REVIEW_NOTE: &str = "\
**Code review skipped** — the content is identical to what the reviewers were \
last handed on this ticket (same commit, same working tree, no changes since), \
so they were not dispatched again. The functional check still ran.\n\n";

/// Static metadata for one participant kind of a verification round. The code
/// reviewers and the functional tester differ only here: the role (which
/// selects the system prompt, tools and shell mode), the roster kind their
/// slots are stored under, the work prompt, the angle supplements and the
/// extraction prompt.
#[derive(Copy, Clone)]
struct Participant {
    role: Role,
    kind: crate::jobs::AgentKind,
    prompt_template: &'static str,
    angles_path: &'static str,
    extraction_prompt_path: &'static str,
}

/// Read-only code reviewers; the cohort size is calibrated per round.
const REVIEWER: Participant = Participant {
    role: Role::Reviewer,
    kind: crate::jobs::AgentKind::Reviewer,
    prompt_template: "review.md",
    angles_path: "review_angles.md",
    extraction_prompt_path: "extraction/reviewer.md",
};

/// The functional tester — the participant that may run the product.
const TESTER: Participant = Participant {
    role: Role::Qa,
    kind: crate::jobs::AgentKind::Tester,
    prompt_template: "qa.md",
    angles_path: "qa_angles.md",
    extraction_prompt_path: "extraction/qa.md",
};

impl Participant {
    /// The round's work prompt with the engineer's response and the round's own
    /// project-commands brief baked in.
    fn prompt(self, ticket: &Ticket, commands_brief: &str) -> String {
        substitute(
            &load_prompt(self.prompt_template),
            &[
                ("{{agent_response}}", engineer_response(ticket)),
                ("{{project_commands}}", commands_brief),
            ],
        )
    }

    /// The round's angle supplements for this participant.
    fn angles(self) -> Vec<String> {
        load_prompt_sections(self.angles_path)
    }

    fn extraction_prompt(self) -> String {
        load_prompt(self.extraction_prompt_path)
    }
}

/// The latest engineer comment on the ticket — the change under verification.
fn engineer_response(ticket: &Ticket) -> &str {
    ticket
        .comments
        .iter()
        .rev()
        .find(|c| c.role == Role::Engineer.as_str())
        .map_or("(no output)", |c| c.content.as_str())
}

/// Run the verification stage on a phase job: build a fresh round, or resume
/// the round recorded in the job's roster.
pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, ACTIVE_PHASE, &job_id).await {
        return;
    }
    // The round's project-commands part — the workspace's discovered diagnostics
    // commands — read once before anything is dispatched: not being able to read
    // them is a technical failure (no bounce budget) — the round cannot be judged
    // without knowing them.
    let commands = match crate::workspace::store().get_diagnostics(&ws.name).await {
        Ok(Some(cmds)) if !cmds.is_empty() => Some(cmds),
        Ok(_) => None,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to load the workspace's project commands — resetting for a fresh attempt",
            );
            reset_phase_attempt(
                &ticket,
                ACTIVE_PHASE,
                &job_id,
                "project commands load failure",
                &format!(
                    "Could not load the workspace's project commands due to a database error: {e}"
                ),
            )
            .await;
            return;
        }
    };
    let Some(roster) = read_roster_or_bail(&ticket.id, &job_id).await else {
        return;
    };
    let resume = !roster.is_empty();
    let plan = if resume {
        let Some(plan) = RoundPlan::from_roster(&roster) else {
            // A roster that is not a verification round (no functional tester
            // row, or a shape another build wrote) is never resumed as one:
            // dropping the job makes the poller re-create it, so the round
            // re-runs whole — with the functional check in it. The job is gone
            // afterwards, so the roster's shape is logged here or not at all.
            warn!(
                ticket = %ticket.id,
                job = %job_id,
                rows = roster.len(),
                kinds = %roster.iter().map(|r| r.kind.as_str()).collect::<Vec<_>>().join(","),
                "Unreadable verification roster — re-driving the round from scratch",
            );
            let _ = crate::jobs::terminalize_job(&crate::session::store().conn, &job_id).await;
            return;
        };
        plan
    } else {
        // A roster write failure leaves the job with no plan: bail and let the
        // poller re-drive it rather than run a round the resume path could
        // misread.
        let Some(plan) = RoundPlan::fresh(&ticket, &ws, &job_id, commands.as_ref()).await else {
            return;
        };
        plan
    };
    execute_round(ticket, ws, job_id, plan, resume, commands).await;
}

/// One verification round's dispatch plan: the two participant cohorts. The
/// reviewer cohort is empty exactly when this round skipped the code review.
struct RoundPlan {
    reviewers: Vec<AgentSlot>,
    testers: Vec<AgentSlot>,
}

impl RoundPlan {
    /// Build a fresh round: calibrated reviewers (none when the content is
    /// already checked) plus the always-present functional tester.
    async fn fresh(
        ticket: &Ticket,
        ws: &Workspace,
        job_id: &str,
        commands: Option<&DiagnosticsCommands>,
    ) -> Option<Self> {
        let commands_brief = commands_brief(commands);
        let tester_prompt = TESTER.prompt(ticket, &commands_brief);
        // The identical-content rule: content already checked on this ticket is
        // not checked again by the reviewers. It never skips the tester.
        let (reviewer_prompt, reviewers) = if skip_review_for_identical_content(ticket, ws).await {
            info!(
                ticket = %ticket.id,
                "Content identical to the reviewed base — dispatching the functional tester only",
            );
            (None, Vec::new())
        } else {
            let prompt = REVIEWER.prompt(ticket, &commands_brief);
            let slots = build_agent_slots(
                &ticket.id,
                REVIEWER.role,
                &prompt,
                &REVIEWER.angles(),
                0,
                compute_reviewer_count(ticket, ws.as_path()).await,
            );
            (Some(prompt), slots)
        };
        // The tester's idx continues the reviewer pack — its value only orders
        // the roster and the Running Agents view, since the cohorts are
        // identified by their roster kind.
        let testers = build_agent_slots(
            &ticket.id,
            TESTER.role,
            &tester_prompt,
            &TESTER.angles(),
            i64::try_from(reviewers.len()).unwrap_or(i64::MAX),
            TESTER_COUNT,
        );

        // The job's stored task records the round's primary work prompt (the
        // phase dispatch re-derives the prompts from live state).
        let task = reviewer_prompt.as_deref().unwrap_or(&tester_prompt);
        sync_phase_job_task(&crate::session::store().conn, job_id, task).await;
        let cohorts = [
            (REVIEWER.kind, reviewers.as_slice()),
            (TESTER.kind, testers.as_slice()),
        ];
        if let Err(e) = insert_round_slots(job_id, &cohorts).await {
            warn!(
                ticket = %ticket.id,
                job = %job_id,
                error = %e,
                "Failed to write the verification roster — round not started",
            );
            return None;
        }
        Some(Self { reviewers, testers })
    }

    /// Split a stored roster back into its cohorts by the rows' own roster
    /// labels — never by slot order, so a slot appended mid-round cannot be
    /// read as the functional tester.
    ///
    /// `None` for a roster this stage cannot read back as a round: it must carry
    /// exactly [`TESTER_COUNT`] tester row (any number of reviewer rows) and no
    /// row of any other kind. Such a roster is never resumed — see [`run`].
    fn from_roster(roster: &[crate::jobs::AgentRow]) -> Option<Self> {
        let cohort = |kind: crate::jobs::AgentKind| -> Vec<AgentSlot> {
            roster
                .iter()
                .filter(|row| row.kind == kind.as_str())
                .map(agent_slot_from_roster_row)
                .collect()
        };
        let reviewers = cohort(REVIEWER.kind);
        let testers = cohort(TESTER.kind);
        if testers.len() != TESTER_COUNT || reviewers.len() + testers.len() != roster.len() {
            return None;
        }
        Some(Self { reviewers, testers })
    }
}

/// The content identity the code reviewers were handed: `HEAD` plus the working
/// tree's identity (see [`crate::git::commands::run_git_worktree_identity`]),
/// read before the round's cohorts run.
///
/// On a resumed round that reading happens at resume time, after whatever the
/// interrupted attempt left in the tree: what the pre-round reading removes from
/// "already reviewed" is a participant's own work, not an interrupted run's
/// leftovers.
struct ReviewBase {
    head: String,
    tree: String,
}

/// The round's project-commands brief for the participants' work prompts: the
/// concrete list of the commands this round runs, in execution order, or the
/// notice that it runs none.
fn commands_brief(commands: Option<&DiagnosticsCommands>) -> String {
    let Some(commands) = commands else {
        return load_prompt("pipeline/round_commands_none.md");
    };
    let list = commands
        .commands()
        .iter()
        .filter_map(|(label, cmd)| cmd.map(|cmd| format!("- {label}: `{cmd}`")))
        .collect::<Vec<_>>()
        .join("\n");
    substitute(
        &load_prompt("pipeline/round_commands.md"),
        &[("{{commands}}", &list)],
    )
}

/// The round's project-commands result: the comment section that records the
/// run (the commands in execution order, what passed, and the command the run
/// stopped at) and whether every command passed — vacuously true for a
/// workspace with no commands configured.
struct ProjectCommands {
    section: String,
    passed: bool,
}

/// Run the workspace's project commands — in order, stopping at the first
/// failure — on the phase body's own task, concurrently with the round's
/// participant cohorts. A command that changes the tree (an auto-formatter, a
/// lint auto-fix) is part of the round exactly like a check: no command is
/// held back to a step of its own.
///
/// The judgement is the exit status and nothing else (`Some(0)` is a pass):
/// what a command writes is kept for the comment — through the output
/// profiles, which trim it for display — but never read to decide the outcome.
/// A check command answering in another language is therefore judged exactly as
/// it is in English (see `profiles`' own note).
async fn run_project_commands(
    commands: Option<&DiagnosticsCommands>,
    ws: &Workspace,
) -> ProjectCommands {
    let Some(commands) = commands else {
        return ProjectCommands {
            section: format!(
                "**Project commands**\n\n{}",
                load_prompt("pipeline/commands_none.md")
            ),
            passed: true,
        };
    };

    let mut lines = String::new();
    let mut failed_at: Option<&str> = None;
    for (label, cmd) in commands.commands() {
        let Some(cmd) = cmd else {
            continue;
        };
        let started = std::time::Instant::now();
        let elapsed = || started.elapsed().as_secs_f64();
        match ShellTool::new(ShellMode::Full)
            .execute_with_status(ws, serde_json::json!({ "command": cmd }))
            .await
        {
            Ok((_output, Some(0))) => {
                let _ = writeln!(lines, "- {label} (`{cmd}`): PASSED in {:.1}s", elapsed());
            }
            Ok((output, _exit_code)) => {
                let display = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                let _ = writeln!(
                    lines,
                    "- {label} (`{cmd}`): FAILED in {:.1}s\n\n```\n{display}\n```",
                    elapsed(),
                );
                failed_at = Some(label);
                break;
            }
            Err(e) => {
                let _ = writeln!(
                    lines,
                    "- {label} (`{cmd}`): FAILED in {:.1}s\n\n```\n{e}\n```",
                    elapsed(),
                );
                failed_at = Some(label);
                break;
            }
        }
    }

    // The commands' shell spills are this round's own leftovers, not an
    // agent's — the phase body runs them with no agent identity.
    crate::tools::shell::cleanup_agent_spills(DIAGNOSTICS_ROLE);
    let footer = match failed_at {
        Some(label) => format!("{} `{label}`", load_prompt("pipeline/commands_failed.md")),
        None => load_prompt("pipeline/commands_passed.md"),
    };
    ProjectCommands {
        section: format!("**Project commands**\n\n{}\n\n{footer}", lines.trim_end()),
        passed: failed_at.is_none(),
    }
}

/// Run the plan's cohorts concurrently on the phase job and finalize the
/// merged round.
async fn execute_round(
    ticket: Arc<Ticket>,
    ws: Workspace,
    job_id: String,
    plan: RoundPlan,
    resume: bool,
    commands: Option<DiagnosticsCommands>,
) {
    if resume {
        // Re-arm the not-Done slots as launched so comment routing and the
        // Running Agents view track the in-flight participants.
        let not_done: Vec<String> = plan
            .reviewers
            .iter()
            .chain(&plan.testers)
            .filter(|s| s.status != crate::jobs::RowStatus::Done)
            .map(|s| s.agent_id.clone())
            .collect();
        if let Err(e) =
            crate::jobs::rearm_roster_launched(&crate::session::store().conn, &job_id, &not_done)
                .await
        {
            warn!(ticket = %ticket.id, job = %job_id, error = %e, "Failed to re-arm resumed verification slots");
        }
    }
    let reviewer_count = plan.reviewers.len();
    let tester_count = plan.testers.len();
    let command_count = commands.as_ref().map_or(0, |c| {
        c.commands().iter().filter(|(_, cmd)| cmd.is_some()).count()
    });
    info!(
        ticket = %ticket.id,
        reviewers = reviewer_count,
        testers = tester_count,
        commands = command_count,
        "Dispatching {reviewer_count} reviewer(s), {tester_count} tester(s) and {command_count} project command(s) in parallel",
    );

    // A skipped code review was never dispatched, so it has neither content to
    // record nor slots to write. The identity is read BEFORE either cohort runs:
    // the round records the content the code review was handed, never the tree
    // the functional tester may have touched by the time the round ends. The
    // reading stages nothing (see `run_git_worktree_identity`); a repository it
    // cannot read leaves no base, which only means a later round reviews the
    // same content again.
    let review_skipped = plan.reviewers.is_empty();
    let base = if review_skipped {
        None
    } else {
        match crate::git::commands::run_git_worktree_identity(ws.as_path()).await {
            Ok((head, tree)) => Some(ReviewBase { head, tree }),
            Err(e) => {
                debug!(error = %e, "Could not read the reviewed content identity");
                None
            }
        }
    };

    // One round: the two participant cohorts and the workspace's project
    // commands, all started at the same time on the same phase job.
    let ((reviewer_results, reviewer_paused), (tester_results, tester_paused), commands_outcome) = tokio::join!(
        dispatch_cohort(&ticket, &ws, REVIEWER, &plan.reviewers, &job_id, resume),
        dispatch_cohort(&ticket, &ws, TESTER, &plan.testers, &job_id, resume),
        run_project_commands(commands.as_ref(), &ws),
    );

    if guard_job_phase(&ticket.id, ACTIVE_PHASE, &job_id).await {
        return;
    }
    // A pause-freeze is NOT a technical failure: leave the job in place for the
    // unpause re-drive (the typed `paused` signal was captured at bail time, so
    // it survives a workspace-unpause race that a live re-read of paused state
    // would miss).
    if reviewer_paused || tester_paused {
        pause_freezing(&ticket, &job_id).await;
        return;
    }
    finalize_round(
        &ws,
        &ticket,
        &reviewer_results,
        &tester_results,
        base,
        review_skipped,
        &job_id,
        &commands_outcome,
    )
    .await;
}

/// Dispatch one participant cohort of the round on the shared phase job. A
/// cohort with no slot (a skipped code review) loads nothing and dispatches
/// nothing — the co-dispatch of the other cohort is what makes the round.
async fn dispatch_cohort(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    participant: Participant,
    slots: &[AgentSlot],
    job_id: &str,
    resume: bool,
) -> (Vec<ParallelVerdict>, bool) {
    if slots.is_empty() {
        return (Vec::new(), false);
    }
    run_parallel_agents(
        ticket,
        ws,
        participant.role,
        &participant.extraction_prompt(),
        ExtractionMode::ScoreVerdict,
        job_id,
        slots,
        ACTIVE_PHASE,
        resume,
    )
    .await
}

/// Finalize the merged round and record the reviewed base when the code review
/// ran.
#[expect(clippy::too_many_arguments)]
async fn finalize_round(
    ws: &Workspace,
    ticket: &Ticket,
    reviewer_results: &[ParallelVerdict],
    tester_results: &[ParallelVerdict],
    base: Option<ReviewBase>,
    review_skipped: bool,
    job_id: &str,
    commands: &ProjectCommands,
) {
    let results: Vec<ParallelVerdict> = reviewer_results
        .iter()
        .chain(tester_results)
        .cloned()
        .collect();
    let transitioned =
        process_round_verdicts(ws, ticket, &results, review_skipped, job_id, commands).await;
    if !review_skipped {
        // Tied to the code review having actually run: the recording is what
        // makes the identical-content rule fire on a later round, and the
        // tester's verdict alone must never stand in for it. A round that did
        // not move the ticket on — the drain abort included — stages nothing.
        record_reviewed_base(ws, &ticket.id, base, transitioned).await;
    }
}

/// Process the merged round's verdicts: build the consolidated comment, decide
/// pass/fail, and update the ticket phase. Returns whether the round's outcome
/// was applied (a pass or a bounce).
async fn process_round_verdicts(
    ws: &Workspace,
    ticket: &Ticket,
    results: &[ParallelVerdict],
    review_skipped: bool,
    job_id: &str,
    commands: &ProjectCommands,
) -> bool {
    // Distinguish the two failure classes: a participant that did NOT complete
    // (NoResponse/ParseFailed) is a HARD TECHNICAL failure — reset the attempt
    // (comment + delete job + pause; no bounce budget). A participant that DID
    // complete but found issues (a Verdict below threshold) is a rework verdict
    // — bounce to development, consuming bounce budget.
    let technical_failure = results.iter().any(ParallelVerdict::is_technical_failure);
    // The round's rework condition is the participants' verdicts OR the project
    // commands: either failing sends the ticket back once per round.
    let rework_failure = !technical_failure
        && (!commands.passed
            || results
                .iter()
                .any(|r| matches!(r, ParallelVerdict::Verdict(v) if !verdict_passes(v))));

    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            stage = STAGE,
            "Verification round cut short by drain — job stays launched for boot resume",
        );
        return false;
    }

    if technical_failure {
        // Hard technical failure: the whole merged round is destroyed and
        // re-driven as a unit (never resumed participant by participant).
        let comment =
            format!("{STAGE} could not complete the round (a participant did not respond).");
        reset_phase_attempt(
            ticket,
            ACTIVE_PHASE,
            job_id,
            "verification failure",
            &comment,
        )
        .await;
        return false;
    }

    // The round's one artifact: the consolidated joint comment, which carries
    // the skipped-code-review note when the reviewers were not dispatched.
    let (round, outcome) = build_round_grouping(
        STAGE,
        results,
        // The stage's own identity: `stage_role(STAGE)` resolves the
        // consolidated comment to the functional tester's badge and icon, so
        // the grouping pass runs and is labelled under the same role.
        Role::Qa,
        // A skipped code review leaves the functional tester as the round's
        // only participant.
        review_skipped,
        ws,
        &ticket.id,
        &ticket.title,
    )
    .await;
    let mut verdict = render_joint_comment(
        &round,
        &outcome,
        &crate::consensus::ItemTable::new(&round.issues),
    );
    if review_skipped {
        verdict.insert_str(0, SKIPPED_REVIEW_NOTE);
    }
    // The round's ONE comment, both halves of its result: the project commands
    // that ran alongside it, then the participants' verdict.
    let comment = format!("{}\n\n---\n\n{verdict}", commands.section);

    if !rework_failure {
        return apply_clean_round(ticket, &comment, job_id).await;
    }

    let outcome = bounce_to_development(
        ticket,
        ACTIVE_PHASE,
        STAGE,
        STAGE,
        ACTIVE_PHASE.as_ref(),
        &comment,
        job_id,
    )
    .await;
    matches!(outcome, FinalizeOutcome::Applied)
}

/// Apply the clean-pass outcome of a verification round: write the joint
/// comment, transition the ticket to sanitation, and delete the phase job.
/// Returns `false` if the transition was not applied (phase moved
/// concurrently).
async fn apply_clean_round(ticket: &Ticket, comment: &str, job_id: &str) -> bool {
    if !matches!(
        comment_and_transition(
            TransitionCtx::buffered(
                ticket,
                ACTIVE_PHASE,
                SUCCESS_PHASE,
                STAGE,
                ACTIVE_PHASE.as_ref(),
            ),
            STAGE,
            comment,
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        return false;
    }
    info!(
        ticket = %ticket.id,
        "{STAGE}: the project commands and every participant passed (≥ {threshold}/10)",
        threshold = VERIFICATION_THRESHOLD,
    );
    // Delete the phase job; the puller creates the next phase job.
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
    true
}

/// Whether a verification verdict passes (score at or above the threshold).
#[must_use]
fn verdict_passes(verdict: &crate::Verdict) -> bool {
    verdict.score >= VERIFICATION_THRESHOLD
}

// ── Reviewer-only git decisions ─────────────────────────────────────────

/// Whether git is usable for the reviewer cohort (a member of the round's
/// participants; the tester never consults it).
async fn git_available(ws: &Workspace) -> bool {
    crate::git::commands::git_is_installed().await
        && crate::git::commands::is_git_repo(ws.as_path())
}

/// Whether this round's code review may be skipped: the working tree is
/// identical to the ticket's recorded reviewed base. A git failure is
/// fail-open — the reviewers run.
async fn skip_review_for_identical_content(ticket: &Ticket, ws: &Workspace) -> bool {
    if !git_available(ws).await {
        return false;
    }
    match compute_review_skip(ticket, ws.as_path()).await {
        Ok(skip) => skip,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Git status check failed for skip-review — dispatching the reviewers",
            );
            false
        }
    }
}

/// Decide whether the reviewer pass may be skipped for a ticket.
fn should_skip_review(
    reviewed_head: Option<&str>,
    reviewed_tree: Option<&str>,
    current_head: Option<&str>,
    current_tree: Option<&str>,
    porcelain: &str,
) -> bool {
    let (Some(base_head), Some(base_tree)) = (reviewed_head, reviewed_tree) else {
        return false;
    };
    let (Some(head), Some(tree)) = (current_head, current_tree) else {
        return false;
    };
    head == base_head && tree == base_tree && !has_unstaged_changes(porcelain)
}

/// Compute the skip-review decision for a ticket.
async fn compute_review_skip(ticket: &Ticket, repo_path: &Path) -> anyhow::Result<bool> {
    let porcelain = run_git_status(repo_path).await?;
    let head = run_git_head(repo_path).await.ok();
    let tree = run_git_write_tree(repo_path).await.ok();
    if (head.is_none() || tree.is_none()) && ticket.reviewed_head.is_some() {
        warn!(
            ticket = %ticket.id,
            head = head.is_some(),
            tree = tree.is_some(),
            "Could not compute full content identity — dispatching the reviewers",
        );
    }
    Ok(should_skip_review(
        ticket.reviewed_head.as_deref(),
        ticket.reviewed_tree.as_deref(),
        head.as_deref(),
        tree.as_deref(),
        &porcelain,
    ))
}

/// Gather the working-tree churn at review dispatch.
async fn working_tree_churn(repo_path: &Path) -> anyhow::Result<i64> {
    let snapshot = run_git_worktree_snapshot(repo_path).await?;
    // An unborn HEAD is not a valid churn baseline: surface as Err so the
    // caller defaults the reviewer base instead of calibrating on a zero diff.
    if snapshot.unborn_head {
        anyhow::bail!("Repository has no commits — no churn baseline");
    }
    // Churn calibration uses only the exact line counts — the
    // huge/binary untracked file-count must NOT influence reviewer counts.
    Ok(snapshot.stats.added + snapshot.stats.removed)
}

/// Compute the reviewer count for a verification round.
///
/// Never fewer than one: the round dispatches a reviewer cohort exactly when the
/// code review runs, and an empty cohort at that point would be indistinguishable
/// from the identical-content skip (which the *plan* decides, not the count).
pub(crate) async fn compute_reviewer_count(ticket: &Ticket, repo_path: &Path) -> usize {
    let tiny = crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_TINY_CHURN;
    let low = crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_LOW_CHURN;
    let high = crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_HIGH_CHURN;

    let count = match working_tree_churn(repo_path).await {
        Ok(total) => {
            let base = crate::pipeline::verdict::review_base_from_signals(total, tiny, low, high);
            // Debug: per-round calibration bookkeeping; an uncomputable churn
            // keeps its warn record below.
            debug!(
                ticket = %ticket.id,
                total_churn = total,
                reviewer_base = base,
                "Reviewer count calibration: base {base} from total churn",
            );
            crate::pipeline::verdict::review_agent_count(base, ticket.priority)
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Could not compute working-tree churn — reviewer base defaults to 3",
            );
            3
        }
    };
    count.max(1)
}

/// Write out the ticket's reviewed base from the identity read before the round
/// ran — the content the code reviewers were handed (see [`ReviewBase`]) — in a
/// round whose code review ran and whose outcome was applied (a pass OR a
/// bounce, which is what makes the identical-content rule fire on the round
/// after a bounced one).
///
/// `git add -A` still runs here, and only here: it is not part of the recording,
/// but the next round's skip check reads HEAD and the *index* tree, so the index
/// has to end a recorded round holding the working tree. It runs under the
/// round's own gate, so a round that never moves on leaves the index and the
/// working tree exactly as it found them — the only trace of a round that
/// records nothing is the unreferenced blobs the pre-dispatch reading wrote, as
/// any content identity does (see
/// [`crate::git::commands::run_git_worktree_identity`]).
async fn record_reviewed_base(
    ws: &Workspace,
    ticket_id: &str,
    base: Option<ReviewBase>,
    transitioned: bool,
) {
    if !transitioned || !git_available(ws).await {
        return;
    }
    // Staged first, recorded second: a round that moves the ticket on leaves the
    // index holding the working tree exactly as before, whether or not the
    // identity could be read.
    if let Err(e) = run_git_add_all(ws.as_path()).await {
        warn!(
            ticket = %ticket_id,
            error = %e,
            "Failed to stage changes after review — reviewed base not recorded",
        );
        return;
    }
    let Some(base) = base else {
        warn!(
            ticket = %ticket_id,
            "Could not read the reviewed content before the round — reviewed base not recorded",
        );
        return;
    };
    if let Err(e) = super::board()
        .set_reviewed_base(ticket_id, Some(&base.head), Some(&base.tree))
        .await
    {
        warn!(
            ticket = %ticket_id,
            error = %e,
            "Failed to record reviewed base — later rounds will re-review",
        );
    } else {
        debug!(ticket = %ticket_id, "Recorded reviewed base after verification");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::commands::MAX_UNTRACKED_SIZE;
    use crate::util::test::init_temp_repo;

    #[tokio::test]
    async fn working_tree_churn_ignores_huge_binary_file_count() {
        let (_dir, repo_path) = init_temp_repo();
        // A normal untracked file (3 lines) plus an oversized untracked file.
        std::fs::write(repo_path.join("a.rs"), b"fn foo() {\n    bar();\n}\n").unwrap();
        let size = usize::try_from(MAX_UNTRACKED_SIZE).unwrap() + 1;
        std::fs::write(repo_path.join("big.bin"), vec![b'a'; size]).unwrap();
        let churn = working_tree_churn(&repo_path).await.unwrap();
        // Churn is added+removed only; the oversized file contributes nothing.
        assert_eq!(churn, 3);
    }
}
