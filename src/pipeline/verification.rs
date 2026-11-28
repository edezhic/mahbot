//! The `verification` stage runs the workspace's project commands, the read-only code
//! reviewers and the single functional tester at the same time on one phase job, and the
//! round produces ONE consolidated comment and one advance/bounce decision. The project
//! commands are nobody's roster slot: they are the round's own work, run in parallel with
//! the participants. A failing command's record carries the part of its output that names the
//! failure ([`failed_command_record`]), so the next round can tell what broke.
//!
//! The round's cohorts are told apart by their stored `agents.kind` labels, never by slot
//! order, and a roster that is not the round's shape is never resumed as one. The
//! identical-content skip, the churn-calibrated reviewer count and the reviewed-base
//! recording are reviewer-only: the tester is never skipped.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use crate::git::commands::{
    has_unstaged_changes, run_git_add_all, run_git_head, run_git_status, run_git_worktree_snapshot,
    run_git_write_tree,
};
use crate::outcome::Outcome;
use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::tools::shell::{ShellMode, ShellTool};
use crate::util::excerpt::failure_excerpt;
use crate::workspace_profile::{Participant, Profile};
use crate::{DiagnosticsCommands, Role, Workspace};

use super::{
    AgentSlot, ExtractionMode, MemberOutcome, ParallelVerdict, TicketPhase, TransitionCtx,
    agent_slot_from_roster_row, bounce_to_development, build_agent_slots, build_round_grouping,
    comment_and_transition, debug, guard_job_phase, info, insert_round_slots, notice,
    pause_freezing, read_roster_or_bail, rearm_resumed_slots, render_joint_comment,
    reset_phase_attempt, run_parallel_agents, sync_phase_job_task, warn,
};

/// The stage's own name: the role the consolidated round comment is authored under, and the
/// label in log fields. (The recorded transition actor is the phase token.)
pub(crate) const STAGE: &str = "Verification";

const ACTIVE_PHASE: TicketPhase = TicketPhase::Verification;

const SUCCESS_PHASE: TicketPhase = TicketPhase::InSanitation;

/// The functional tester runs exactly one agent per round — the round's functional check,
/// which no round may run without.
const TESTER_COUNT: usize = 1;

/// Rendered at the head of the consolidated comment of a round whose reviewer cohort was not
/// dispatched: the skipped code review is part of the round's result. It ends with the blank
/// line that separates it from the round's own sections.
const SKIPPED_REVIEW_NOTE: &str = "\
**Code review skipped** — the content is identical to what the reviewers were \
last handed on this ticket (same commit, same working tree, no changes since), \
so they were not dispatched again. The functional check still ran.\n\n";

/// Rendered instead of [`SKIPPED_REVIEW_NOTE`] when the workspace's profile does not staff the
/// verification stage with reviewers.
const PROFILE_NO_REVIEW_NOTE: &str = "\
**Code review not part of this workspace's profile** — the functional check still ran.\n\n";

/// Static metadata for one participant cohort of a verification round — the code reviewers or the
/// functional tester. A cohort differs from the other only here: the role (which selects the
/// system prompt, tools and shell mode), the roster kind its slots are stored under, the work
/// prompt, the angle supplements and the extraction prompt.
#[derive(Copy, Clone)]
struct Cohort {
    role: Role,
    kind: crate::jobs::AgentKind,
    prompt_template: &'static str,
    angles_path: &'static str,
    extraction_prompt_path: &'static str,
}

/// Read-only code reviewers; the cohort size is calibrated per round.
const REVIEWER: Cohort = Cohort {
    role: Role::Reviewer,
    kind: crate::jobs::AgentKind::Reviewer,
    prompt_template: "review.md",
    angles_path: "review_angles.md",
    extraction_prompt_path: "extraction/reviewer.md",
};

/// The functional tester — the participant that may run the product.
const TESTER: Cohort = Cohort {
    role: Role::Qa,
    kind: crate::jobs::AgentKind::Tester,
    prompt_template: "qa.md",
    angles_path: "qa_angles.md",
    extraction_prompt_path: "extraction/qa.md",
};

impl Cohort {
    /// The round's work prompt with the engineer's response and the project-commands brief
    /// baked in.
    fn prompt(self, ticket: &Ticket, commands_brief: &str) -> String {
        substitute(
            &load_prompt(self.prompt_template),
            &[
                ("{{agent_response}}", engineer_response(ticket)),
                ("{{project_commands}}", commands_brief),
            ],
        )
    }

    fn angles(self) -> Vec<String> {
        load_prompt_sections(self.angles_path)
    }

    fn extraction_prompt(self) -> String {
        load_prompt(self.extraction_prompt_path)
    }

    /// This participant's name in the round: the tester is its only member, the reviewers are
    /// numbered by their slot.
    fn label(self, slot: &AgentSlot) -> String {
        match self.kind {
            crate::jobs::AgentKind::Tester => "functional tester".to_string(),
            _ => format!("code reviewer {}", slot.idx + 1),
        }
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

/// The profile a fresh round runs on, recorded on the job row so a resume finishes on the settings
/// it began with. Best-effort: a failed write is logged and the round runs on the profile anyway.
async fn record_round_profile(ws: &Workspace, ticket: &Ticket, job_id: &str) -> Profile {
    let live = crate::workspace_profile::read(ws).profile;
    if let Err(e) = crate::jobs::set_round_profile(job_id, &live.to_json()).await {
        warn!(
            ticket = %ticket.id,
            job = %job_id,
            error = %e,
            "Failed to record the round's profile — the round runs on it anyway",
        );
    }
    live
}

/// The profile a resumed round finishes on: the one it recorded when it started. Best-effort: a
/// failed read or an unreadable record leaves it on the live profile, which is what it would have
/// used anyway.
async fn resumed_round_profile(ws: &Workspace, ticket: &Ticket, job_id: &str) -> Profile {
    let live = crate::workspace_profile::read(ws).profile;
    match crate::jobs::round_profile(job_id).await {
        Ok(Some(recorded)) => crate::workspace_profile::Profile::from_json(&recorded)
            .unwrap_or_else(|e| {
                warn!(
                    ticket = %ticket.id,
                    job = %job_id,
                    error = %e,
                    "Unreadable round profile — finishing the round on the live profile",
                );
                live
            }),
        Ok(None) => live,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                job = %job_id,
                error = %e,
                "Failed to read the round's profile — finishing it on the live profile",
            );
            live
        }
    }
}

/// Run the verification stage on a phase job: build a fresh round, or resume the round
/// recorded in the job's roster.
pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, ACTIVE_PHASE, &job_id).await {
        return;
    }
    // The round's project-commands part — the workspace's discovered diagnostics commands — read
    // once before anything is dispatched: not being able to read them is a technical failure (no
    // bounce budget) — the round cannot be judged without knowing them.
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
                Some(notice::Cause::ProjectCommands {
                    terms: notice::Terms::plain(crate::util::failure_detail(
                        &format!("{e:#}"),
                        "project commands load failure",
                    )),
                }),
            )
            .await;
            return;
        }
    };
    let Some(roster) = read_roster_or_bail(&ticket.id, &job_id).await else {
        return;
    };
    let resume = !roster.is_empty();
    let profile = if resume {
        resumed_round_profile(&ws, &ticket, &job_id).await
    } else {
        record_round_profile(&ws, &ticket, &job_id).await
    };
    let plan = if resume {
        let Some(plan) = RoundPlan::from_roster(&roster) else {
            // A roster that is not a verification round is never resumed as one: dropping the
            // job makes the poller re-create it, so the round re-runs whole — with the
            // functional check in it. The job is gone afterwards, so the roster's shape is
            // logged here or not at all.
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
        // A roster write failure leaves the job with no plan: bail and let the poller re-drive
        // it rather than run a round the resume path could misread.
        let Some(plan) = RoundPlan::fresh(&ticket, &ws, &job_id, commands.as_ref(), &profile).await
        else {
            return;
        };
        plan
    };
    execute_round(ticket, ws, job_id, plan, resume, commands, profile).await;
}

/// One verification round's dispatch plan: the two participant cohorts. The reviewer cohort
/// is empty exactly when this round skipped the code review.
struct RoundPlan {
    reviewers: Vec<AgentSlot>,
    testers: Vec<AgentSlot>,
}

impl RoundPlan {
    /// Build a fresh round: calibrated reviewers (none when the content is already checked)
    /// plus the always-present functional tester.
    async fn fresh(
        ticket: &Ticket,
        ws: &Workspace,
        job_id: &str,
        commands: Option<&DiagnosticsCommands>,
        profile: &Profile,
    ) -> Option<Self> {
        let commands_brief = commands_brief(commands);
        let tester_prompt = TESTER.prompt(ticket, &commands_brief);
        // The profile decides whether reviewers staff the stage; the identical-content rule
        // decides whether they run this round. Neither ever skips the tester.
        let (reviewer_prompt, reviewers) = if !profile
            .includes(TicketPhase::Verification, Participant::Reviewer)
        {
            info!(
                ticket = %ticket.id,
                "The workspace profile staffs verification without reviewers — dispatching the \
                 functional tester only",
            );
            (None, Vec::new())
        } else if skip_review_for_identical_content(ticket, ws).await {
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
        // The tester's idx continues the reviewer pack — its value only orders the roster and
        // the Running Agents view; cohorts are identified by roster kind.
        let testers = build_agent_slots(
            &ticket.id,
            TESTER.role,
            &tester_prompt,
            &TESTER.angles(),
            i64::try_from(reviewers.len()).unwrap_or(i64::MAX),
            TESTER_COUNT,
        );

        // The job's stored task records the round's primary work prompt (the phase dispatch
        // re-derives prompts from live state).
        let task = reviewer_prompt.as_deref().unwrap_or(&tester_prompt);
        sync_phase_job_task(&crate::session::store().conn, &ticket.id, job_id, task).await;
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

    /// Split a stored roster back into its cohorts by the rows' own roster labels — never by
    /// slot order, so a slot appended mid-round cannot be read as the functional tester.
    ///
    /// `None` for a roster this stage cannot read back as a round: it must carry exactly
    /// [`TESTER_COUNT`] tester row (any number of reviewer rows — a resumed round replays the
    /// cohort it wrote) and no row of any other kind. Such a roster is never resumed.
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

/// The content identity the code reviewers were handed: `HEAD` plus the working tree's
/// identity ([`crate::git::commands::run_git_worktree_identity`]), read before the round's
/// cohorts run.
///
/// On a resumed round that reading happens at resume time, after whatever the interrupted
/// attempt left in the tree: what the pre-round reading removes from "already reviewed" is a
/// participant's own work, not an interrupted run's leftovers.
struct ReviewBase {
    head: String,
    tree: String,
}

/// The round's project-commands brief for the participants' work prompts: the concrete list of
/// commands this round runs, in execution order, or the notice that it runs none.
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

/// The round's project-commands result: the comment section that records the run (commands in
/// execution order, what passed, and the command it stopped at) and whether every command
/// passed — vacuously true with no commands configured.
struct ProjectCommands {
    section: String,
    passed: bool,
}

/// Run the workspace's project commands — in order, stopping at the first failure — on the
/// phase body's own task, concurrently with the participant cohorts. A command that changes the
/// tree (an auto-formatter, a lint auto-fix) is part of the round like a check: no command is
/// held back to a step of its own.
///
/// The judgement is the exit status and nothing else (`Some(0)` is a pass): what a command
/// writes is kept for the comment, but never read to decide the outcome. A failing command's
/// output is read unfiltered, so the record is built from what the tool printed rather than from
/// a profile's positional reading of it — [`failed_command_record`] keeps the lines that name
/// the failure, wherever they are in the output.
///
/// These commands write the folder outside any agent, so the freeze reaches them here: the pause is
/// read live before each one, and a frozen workspace stops the round instead of running the rest.
/// `Ok(None)` is that stop; `Err` is a pause that could not be read, which is no freeze at all.
async fn run_project_commands(
    commands: Option<&DiagnosticsCommands>,
    ws: &Workspace,
) -> anyhow::Result<Option<ProjectCommands>> {
    let Some(commands) = commands else {
        return Ok(Some(ProjectCommands {
            section: format!(
                "**Project commands**\n\n{}",
                load_prompt("pipeline/commands_none.md")
            ),
            passed: true,
        }));
    };

    let mut lines = String::new();
    let mut failed_at: Option<&str> = None;
    for (label, cmd) in commands.commands() {
        let Some(cmd) = cmd else {
            continue;
        };
        if crate::workspace::paused_now(&ws.name).await? {
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let elapsed = || started.elapsed().as_secs_f64();
        match ShellTool::new(ShellMode::Full)
            .unfiltered()
            .execute_with_status(ws, serde_json::json!({ "command": cmd }))
            .await
        {
            Ok((_output, Some(0))) => {
                let _ = writeln!(lines, "- {label} (`{cmd}`): PASSED in {:.1}s", elapsed());
            }
            Ok((output, exit_code)) => {
                failed_command_record(&mut lines, label, cmd, elapsed(), exit_code, &output);
                failed_at = Some(label);
                break;
            }
            // A timeout or a launch failure: the tool's message carries the captured output's
            // failure-identifying excerpt, so the record needs no second cut here.
            Err(e) => {
                write_failed(&mut lines, label, cmd, elapsed(), "", &e.to_string());
                failed_at = Some(label);
                break;
            }
        }
    }

    let footer = match failed_at {
        Some(label) => format!("{} `{label}`", load_prompt("pipeline/commands_failed.md")),
        None => load_prompt("pipeline/commands_passed.md"),
    };
    Ok(Some(ProjectCommands {
        section: format!("**Project commands**\n\n{}\n\n{footer}", lines.trim_end()),
        passed: failed_at.is_none(),
    }))
}

/// The record of one failed command: the exit status on the command's own line, then the part of
/// its output that names the failure — inline, because the record must stand on its own. Output
/// that fits is recorded whole; a longer one is cut to the lines that identify the failure and
/// their context ([`failure_excerpt`]), never to a positional head and tail that a middle failure
/// would not survive.
fn failed_command_record(
    lines: &mut String,
    label: &str,
    cmd: &str,
    elapsed: f64,
    exit_code: Option<i32>,
    output: &str,
) {
    let suffix = exit_code.map_or_else(
        || " (terminated by signal)".to_string(),
        |code| format!(" (exit status {code})"),
    );
    let output = output.trim();
    let display = if output.is_empty() {
        "(no output)".to_string()
    } else {
        failure_excerpt(output)
    };
    write_failed(lines, label, cmd, elapsed, &suffix, &display);
}

/// Write a failed command's record: its verdict line (`suffix` naming how it failed), then its text
/// in a fence no run of backticks in that text can close.
fn write_failed(
    lines: &mut String,
    label: &str,
    cmd: &str,
    elapsed: f64,
    suffix: &str,
    text: &str,
) {
    let longest = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    let fence = "`".repeat((longest + 1).max(3));
    let _ = writeln!(
        lines,
        "- {label} (`{cmd}`): FAILED in {elapsed:.1}s{suffix}\n\n{fence}\n{text}\n{fence}"
    );
}

async fn execute_round(
    ticket: Arc<Ticket>,
    ws: Workspace,
    job_id: String,
    plan: RoundPlan,
    resume: bool,
    commands: Option<DiagnosticsCommands>,
    profile: Profile,
) {
    if resume {
        // A fresh round's slots are already launched — re-arming them would only stamp the
        // job's updated time again.
        let cohorts: [&[AgentSlot]; 2] = [&plan.reviewers, &plan.testers];
        rearm_resumed_slots(&ticket.id, &job_id, "verification", &cohorts).await;
    }
    let reviewer_count = plan.reviewers.len();
    let tester_count = plan.testers.len();
    let command_count = commands.as_ref().map_or(0, |c| {
        c.commands().iter().filter(|(_, cmd)| cmd.is_some()).count()
    });
    info!(
        ticket = %ticket.id,
        "Dispatching {reviewer_count} reviewer(s), {tester_count} tester(s) and {command_count} project command(s) in parallel",
    );

    // A skipped code review was never dispatched, so it has neither content to record nor slots
    // to write. The identity is read BEFORE either cohort runs: the round records the content
    // the code review was handed, never the tree the tester may have touched by the end. A
    // repository the reading cannot read leaves no base, which only means a later round reviews
    // the same content again.
    let review_note = plan.reviewers.is_empty().then(|| {
        if profile.includes(TicketPhase::Verification, Participant::Reviewer) {
            SKIPPED_REVIEW_NOTE
        } else {
            PROFILE_NO_REVIEW_NOTE
        }
    });
    let base = if review_note.is_some() {
        None
    } else {
        match crate::git::commands::run_git_worktree_identity(ws.as_path()).await {
            Ok((head, tree)) => Some(ReviewBase { head, tree }),
            Err(e) => {
                debug!(
                    ticket = %ticket.id,
                    error = %e,
                    "Could not read the reviewed content identity",
                );
                None
            }
        }
    };

    // One round: the two participant cohorts and the project commands, all started at the same
    // time on the same phase job.
    let ((reviewer_results, reviewer_paused), (tester_results, tester_paused), commands) = tokio::join!(
        dispatch_cohort(&ticket, &ws, REVIEWER, &plan.reviewers, &job_id, resume),
        dispatch_cohort(&ticket, &ws, TESTER, &plan.testers, &job_id, resume),
        run_project_commands(commands.as_ref(), &ws),
    );

    if guard_job_phase(&ticket.id, ACTIVE_PHASE, &job_id).await {
        return;
    }
    // A pause-freeze is NOT a technical failure: leave the job in place for the unpause re-drive.
    // Each of the three sources read the pause live by itself — the cohorts at their run start, the
    // project commands before each command — so what is handled here is the freeze in force when
    // the work stopped, not the one seen when it was decided.
    let frozen = reviewer_paused || tester_paused;
    let commands = match commands {
        // A pause that cannot be read is not one: nothing would ever unpause this workspace, so the
        // round is reset for a fresh attempt instead of being parked for a re-drive that cannot come.
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Could not read the workspace's pause — resetting for a fresh attempt",
            );
            reset_phase_attempt(&ticket, ACTIVE_PHASE, &job_id, None).await;
            return;
        }
        Ok(Some(commands)) if !frozen => commands,
        // A cohort's freeze, or the project commands': the round returns with its job in place.
        Ok(_) => {
            pause_freezing(&ticket, &job_id).await;
            return;
        }
    };
    finalize_round(
        &ws,
        &ticket,
        &reviewer_results,
        &tester_results,
        base,
        review_note,
        &job_id,
        &commands,
        &profile,
    )
    .await;
}

/// One participant's outcome in the round, named as the round reports it — the identity a stage
/// death notice needs, which the merged verdict list does not carry.
type Labelled = (String, MemberOutcome);

/// Dispatch one participant cohort on the shared phase job, named per slot. A cohort with no slot
/// (a skipped code review) loads nothing and dispatches nothing.
async fn dispatch_cohort(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    cohort: Cohort,
    slots: &[AgentSlot],
    job_id: &str,
    resume: bool,
) -> (Vec<Labelled>, bool) {
    if slots.is_empty() {
        return (Vec::new(), false);
    }
    let (outcomes, paused) = run_parallel_agents(
        ticket,
        ws,
        cohort.role,
        &cohort.extraction_prompt(),
        ExtractionMode::ScoreVerdict,
        job_id,
        slots,
        ACTIVE_PHASE,
        resume,
    )
    .await;
    // The results come back in slot order, so a label and its outcome stay paired.
    let named = slots
        .iter()
        .zip(outcomes)
        .map(|(slot, outcome)| (cohort.label(slot), outcome))
        .collect();
    (named, paused)
}

#[expect(clippy::too_many_arguments)]
async fn finalize_round(
    ws: &Workspace,
    ticket: &Ticket,
    reviewer_results: &[Labelled],
    tester_results: &[Labelled],
    base: Option<ReviewBase>,
    review_note: Option<&'static str>,
    job_id: &str,
    commands: &ProjectCommands,
    profile: &Profile,
) {
    let results: Vec<Labelled> = reviewer_results
        .iter()
        .chain(tester_results)
        .cloned()
        .collect();
    let transitioned =
        process_round_verdicts(ws, ticket, &results, review_note, job_id, commands, profile).await;
    if review_note.is_none() {
        // Tied to the code review having actually run: the recording is what makes the
        // identical-content rule fire later, and the tester's verdict must never stand in for
        // it. A round that did not move the ticket on stages nothing.
        record_reviewed_base(ws, &ticket.id, base, transitioned).await;
    }
}

/// Process the merged round's verdicts: build the consolidated comment, decide pass/fail, and
/// update the ticket phase. Returns whether the outcome was applied.
async fn process_round_verdicts(
    ws: &Workspace,
    ticket: &Ticket,
    results: &[Labelled],
    review_note: Option<&'static str>,
    job_id: &str,
    commands: &ProjectCommands,
    profile: &Profile,
) -> bool {
    // Two failure classes: a participant that did NOT complete (NoResponse/ParseFailed) is a
    // HARD TECHNICAL failure — reset the attempt (delete job + pause; no bounce budget). One
    // that completed but found issues is a rework verdict — bounce to development, consuming
    // bounce budget.
    let technical_failure = results
        .iter()
        .any(|(_, outcome)| outcome.verdict.is_technical_failure());
    let verdicts: Vec<ParallelVerdict> = results
        .iter()
        .map(|(_, outcome)| outcome.verdict.clone())
        .collect();
    // The rework condition is the participants' verdicts OR the project commands.
    let rework_failure = !technical_failure && round_fails(commands.passed, &verdicts, profile);

    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            stage = STAGE,
            "Verification round cut short by drain — job stays launched for boot resume",
        );
        return false;
    }

    if technical_failure {
        // Hard technical failure: the whole merged round is destroyed and re-driven as a unit.
        warn!(
            ticket = %ticket.id,
            stage = STAGE,
            "Verification round hit a technical participant failure — resetting for a fresh attempt",
        );
        // One notice per stage death, naming every participant that failed. A round whose only
        // failures were cancellations or bails reports nothing: those are not deaths.
        let failed: Vec<notice::Member> = results
            .iter()
            .filter_map(|(label, outcome)| notice::Member::of(label, outcome))
            .collect();
        let death = (!failed.is_empty()).then_some(notice::Cause::Participants(failed));
        reset_phase_attempt(ticket, ACTIVE_PHASE, job_id, death).await;
        return false;
    }

    // The round's one artifact: the consolidated joint comment, carrying the
    // skipped-code-review note when the reviewers were not dispatched.
    let (round, outcome) = build_round_grouping(
        STAGE,
        &verdicts,
        // `stage_role(STAGE)` resolves the comment to the tester's badge and icon, so the
        // grouping pass runs and is labelled under the same role.
        Role::Qa,
        // A skipped code review leaves the tester as the round's only participant.
        review_note.is_some(),
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
    if let Some(note) = review_note {
        verdict.insert_str(0, note);
    }
    // The round's ONE comment, both halves of its result: the project commands, then the
    // participants' verdict.
    let comment = format!("{}\n\n---\n\n{verdict}", commands.section);

    if !rework_failure {
        return apply_clean_round(ticket, &comment, job_id, profile).await;
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
    matches!(outcome, Outcome::Done)
}

/// Apply the clean-pass outcome: write the joint comment, transition the ticket to sanitation,
/// and delete the phase job. `false` if the transition was not applied (phase moved concurrently).
async fn apply_clean_round(
    ticket: &Ticket,
    comment: &str,
    job_id: &str,
    profile: &Profile,
) -> bool {
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
        Outcome::Done
    ) {
        return false;
    }
    info!(
        ticket = %ticket.id,
        threshold = profile.pass_threshold,
        every_check = profile.require_all_checks,
        "{STAGE}: passed",
    );
    // Delete the phase job; the puller creates the next phase job.
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
    true
}

/// Whether the round must bounce: a project command that failed where the profile requires every
/// check to pass, or a participant scoring under the profile's threshold.
fn round_fails(commands_passed: bool, verdicts: &[ParallelVerdict], profile: &Profile) -> bool {
    (profile.require_all_checks && !commands_passed)
        || verdicts.iter().any(|verdict| {
            matches!(verdict, ParallelVerdict::Verdict(v) if v.score < profile.pass_threshold)
        })
}

/// Whether git is usable for the reviewer cohort; the tester never consults it.
async fn git_available(ws: &Workspace) -> bool {
    crate::git::commands::git_is_installed().await
        && crate::git::commands::is_git_repo(ws.as_path())
}

/// Whether this round's code review may be skipped: the working tree is identical to the
/// ticket's recorded reviewed base. A git failure is fail-open — the reviewers run.
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

async fn working_tree_churn(repo_path: &Path) -> anyhow::Result<i64> {
    let snapshot = run_git_worktree_snapshot(repo_path).await?;
    // An unborn HEAD is not a valid churn baseline: surface as Err so the caller defaults the
    // reviewer base instead of calibrating on a zero diff.
    if snapshot.unborn_head {
        anyhow::bail!("Repository has no commits — no churn baseline");
    }
    // Churn calibration uses only the exact line counts — the huge/binary untracked file-count
    // must NOT influence reviewer counts.
    Ok(snapshot.stats.added + snapshot.stats.removed)
}

/// Compute the reviewer count for a round: the working tree's churn calibrates the band (1 up
/// to [`crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_LOW_CHURN`] changed lines, 2 below
/// [`crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_HIGH_CHURN`], and
/// [`crate::pipeline::verdict::MAX_REVIEWERS`] from there), the priority floor applies on top,
/// and an unmeasurable churn falls back to the ceiling.
///
/// Every path stays between one reviewer and the ceiling — an empty cohort would be
/// indistinguishable from the identical-content skip, which the *plan* decides.
pub(crate) async fn compute_reviewer_count(ticket: &Ticket, repo_path: &Path) -> usize {
    use crate::pipeline::verdict::{
        DEFAULT_REVIEW_COUNT_HIGH_CHURN, DEFAULT_REVIEW_COUNT_LOW_CHURN, MAX_REVIEWERS,
        review_agent_count, review_base_from_signals,
    };

    let base = match working_tree_churn(repo_path).await {
        Ok(total) => {
            let base = review_base_from_signals(
                total,
                DEFAULT_REVIEW_COUNT_LOW_CHURN,
                DEFAULT_REVIEW_COUNT_HIGH_CHURN,
            );
            // Debug: per-round calibration bookkeeping; an uncomputable churn keeps its warn
            // record below.
            debug!(
                ticket = %ticket.id,
                total_churn = total,
                reviewer_base = base,
                "Reviewer count calibration: base {base} from total churn",
            );
            base
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Could not compute working-tree churn — reviewer base defaults to the ceiling",
            );
            MAX_REVIEWERS
        }
    };
    review_agent_count(base, ticket.priority)
}

/// Write out the ticket's reviewed base from the identity read before the round ran — the
/// content the code reviewers were handed (see [`ReviewBase`]) — in a round whose code review
/// ran and whose outcome was applied (a pass OR a bounce, which is what makes the
/// identical-content rule fire on the round after a bounced one).
///
/// `git add -A` still runs here, and only here: it is not part of the recording, but the next
/// round's skip check reads HEAD and the *index* tree, so the index has to end a recorded round
/// holding the working tree. It runs under the round's own gate, so a round that never moves on
/// leaves the index and the working tree as it found them.
async fn record_reviewed_base(
    ws: &Workspace,
    ticket_id: &str,
    base: Option<ReviewBase>,
    transitioned: bool,
) {
    if !transitioned || !git_available(ws).await {
        return;
    }
    // Staged first, recorded second: a round that moves the ticket on leaves the index holding
    // the working tree exactly as before, whether or not the identity could be read.
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
        std::fs::write(repo_path.join("a.rs"), b"fn foo() {\n    bar();\n}\n").unwrap();
        let size = usize::try_from(MAX_UNTRACKED_SIZE).unwrap() + 1;
        std::fs::write(repo_path.join("big.bin"), vec![b'a'; size]).unwrap();
        let churn = working_tree_churn(&repo_path).await.unwrap();
        // Churn is added+removed only; the oversized file contributes nothing.
        assert_eq!(churn, 3);
    }

    /// A command that fails after ~28 KB of noise: the record must carry the lines that name the
    /// failure, wherever they are, and never the positional head a volume-based cut would show.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_failing_project_command_records_the_lines_that_name_the_failure() {
        let dir = tempfile::TempDir::new().unwrap();
        let ws = crate::workspace::test_ws(dir.path());
        std::fs::write(
            dir.path().join("fail.sh"),
            "for i in $(seq 1 800); do echo \"checking item $i of the suite\"; done\n\
             echo '---- test_add stdout ----'\n\
             echo \"thread 'test_add' panicked at src/lib.rs:10:5:\"\n\
             echo 'assertion `left == right` failed'\n\
             for i in $(seq 1 400); do echo \"cleanup note $i\"; done\n\
             exit 101\n",
        )
        .unwrap();
        let commands = DiagnosticsCommands {
            unit_test: Some("sh fail.sh".to_string()),
            ..Default::default()
        };

        let record = run_project_commands(Some(&commands), &ws)
            .await
            .expect("the pause was read")
            .expect("the round ran");

        assert!(!record.passed);
        assert!(
            record.section.contains("(exit status 101)"),
            "{}",
            record.section
        );
        for needle in [
            "---- test_add stdout ----",
            "thread 'test_add' panicked at src/lib.rs:10:5:",
            "assertion `left == right` failed",
        ] {
            assert!(
                record.section.contains(needle),
                "the record lacks {needle:?}:\n{}",
                record.section
            );
        }
        assert!(
            !record.section.contains("checking item 1 of the suite"),
            "the record must not spend its room on the output's head:\n{}",
            record.section
        );
        assert!(
            record.section.contains("lines omitted"),
            "a partial record says so:\n{}",
            record.section
        );
    }

    /// A passing command's record is exactly what it always was: its own line, nothing of its
    /// output.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_passing_project_command_records_only_its_verdict() {
        let dir = tempfile::TempDir::new().unwrap();
        let ws = crate::workspace::test_ws(dir.path());
        std::fs::write(
            dir.path().join("pass.sh"),
            "echo passing-output-marker\nexit 0\n",
        )
        .unwrap();
        let commands = DiagnosticsCommands {
            format: Some("sh pass.sh".to_string()),
            ..Default::default()
        };

        let record = run_project_commands(Some(&commands), &ws)
            .await
            .expect("the pause was read")
            .expect("the round ran");

        assert!(record.passed);
        assert!(
            record.section.contains("format (`sh pass.sh`): PASSED in "),
            "{}",
            record.section
        );
        assert!(
            !record.section.contains("passing-output-marker"),
            "{}",
            record.section
        );
    }

    /// A command's own output cannot end the record's block: the fence grows past its backticks.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_record_fences_output_that_holds_backticks() {
        let dir = tempfile::TempDir::new().unwrap();
        let ws = crate::workspace::test_ws(dir.path());
        std::fs::write(
            dir.path().join("fence.sh"),
            "printf 'error: a block\\n```\\ninside the output\\n```\\n'\nexit 1\n",
        )
        .unwrap();
        let commands = DiagnosticsCommands {
            unit_test: Some("sh fence.sh".to_string()),
            ..Default::default()
        };

        let record = run_project_commands(Some(&commands), &ws)
            .await
            .expect("the pause was read")
            .expect("the round ran");

        assert!(record.section.contains("````"), "{}", record.section);
        assert!(
            record.section.contains("inside the output"),
            "{}",
            record.section
        );
    }

    /// A command that printed nothing still records how it failed.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_silent_failure_records_its_exit_status() {
        let dir = tempfile::TempDir::new().unwrap();
        let ws = crate::workspace::test_ws(dir.path());
        let commands = DiagnosticsCommands {
            unit_test: Some("exit 1".to_string()),
            ..Default::default()
        };

        let record = run_project_commands(Some(&commands), &ws)
            .await
            .expect("the pause was read")
            .expect("the round ran");

        assert!(!record.passed);
        assert!(
            record.section.contains("(exit status 1)"),
            "{}",
            record.section
        );
        assert!(record.section.contains("(no output)"), "{}", record.section);
    }

    /// A freeze stops the project commands before the first one runs: the round is left for the
    /// unpause re-drive, so nothing was written and there is no record to render.
    #[cfg(unix)]
    #[serial_test::serial]
    #[tokio::test]
    async fn a_frozen_workspace_runs_no_project_command() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;
        let dir = tempfile::TempDir::new().unwrap();
        let ws = crate::util::test::create_test_workspace(
            dir.path().to_str().unwrap(),
            "frozen_commands_ws",
        )
        .await;
        crate::workspace::store()
            .set_paused(&ws.name, true)
            .await
            .expect("freeze the workspace");
        let marker = dir.path().join("ran");
        let commands = DiagnosticsCommands {
            unit_test: Some(format!("touch {}", marker.display())),
            ..Default::default()
        };

        assert!(
            run_project_commands(Some(&commands), &ws)
                .await
                .expect("the pause was read")
                .is_none(),
            "a frozen workspace runs no command"
        );
        assert!(!marker.exists(), "and nothing was written to the folder");
    }

    /// The profile's `require_all_checks` decides whether a failed project command bounces the
    /// round: the default requires it, a profile that pins `false` does not.
    #[test]
    fn a_failed_command_bounces_only_when_the_profile_requires_every_check() {
        let lenient = Profile::resolve(Some(r#"{"require_all_checks":false}"#)).unwrap();
        assert!(
            !round_fails(false, &[], &lenient),
            "a profile that does not require every check lets a failed command pass the round"
        );
        assert!(
            round_fails(false, &[], &Profile::default()),
            "the default profile bounces a round whose project command failed"
        );
    }

    /// The profile's `pass_threshold` is the round's bar: a participant at it passes, one under it
    /// bounces the round.
    #[test]
    fn a_verdict_at_the_profiles_threshold_passes_and_one_below_it_fails() {
        let profile = Profile::resolve(Some(r#"{"pass_threshold":7}"#)).unwrap();
        let at_threshold = ParallelVerdict::Verdict(crate::Verdict {
            score: 7,
            issues_detected: Vec::new(),
        });
        let below = ParallelVerdict::Verdict(crate::Verdict {
            score: 6,
            issues_detected: Vec::new(),
        });

        assert!(!round_fails(true, &[at_threshold], &profile));
        assert!(round_fails(true, &[below], &profile));
    }

    /// A profile may staff verification without reviewers: the round then runs the functional
    /// tester alone, which the stage reads off the profile rather than off a missing roster.
    #[test]
    fn a_profile_can_staff_verification_without_reviewers() {
        let profile = Profile::resolve(Some(r#"{"stages":{"verification":["tester"]}}"#)).unwrap();
        assert!(!profile.includes(TicketPhase::Verification, Participant::Reviewer));
        assert!(profile.includes(TicketPhase::Verification, Participant::Tester));
    }

    /// A resumed round finishes on the profile it started with: the record on its job row wins over
    /// whatever the workspace carries now, so a mid-round edit cannot change its pass conditions.
    #[tokio::test]
    #[serial_test::serial]
    async fn a_resumed_round_reads_the_profile_it_recorded() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("round_profile")
                .to_string_lossy(),
            "round_profile_ws",
        )
        .await;
        let board = crate::pipeline::board::store();
        let id =
            crate::util::test::make_ticket(board, &ws, "A ticket", TicketPhase::Verification).await;
        let ticket = crate::util::test::expect_ticket(board, &id).await;
        let job_id = &format!("round_profile_job_{}", crate::generate_suffix());
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            job_id,
            "verification",
            "reviewer",
            &ws.name,
        )
        .timestamps(crate::db::now())
        .insert()
        .await
        .expect("seed the round's job row");

        // The round starts under one profile and the workspace is edited to another mid-round.
        let started = Profile::resolve(Some(r#"{"pass_threshold":6,"require_all_checks":false}"#))
            .expect("a legal profile");
        crate::jobs::set_round_profile(job_id, &started.to_json())
            .await
            .expect("record the round's profile");
        crate::workspace::store()
            .set_profile(
                &ws.name,
                Some(r#"{"pass_threshold":10,"require_all_checks":true}"#),
            )
            .await
            .expect("edit the workspace's profile mid-round");
        let edited = crate::workspace::store()
            .get_by_name(&ws.name)
            .await
            .expect("read the workspace")
            .expect("the workspace exists");

        assert_eq!(
            resumed_round_profile(&edited, &ticket, job_id).await,
            started,
            "the resumed round finishes on the pass conditions it began with"
        );
        assert_eq!(
            record_round_profile(&edited, &ticket, job_id).await,
            Profile::resolve(edited.profile.as_deref()).expect("the live profile"),
            "a fresh round reads the live profile"
        );
    }

    /// The profile's reviewer roster reaches the plan: a reviewer-less profile dispatches the
    /// functional tester alone, and the roster rows say so.
    #[tokio::test]
    #[serial_test::serial]
    async fn a_reviewerless_profile_dispatches_the_tester_alone() {
        let _store = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_management_test_stores().await;
        let ws = crate::util::test::create_test_workspace(
            &crate::util::test::test_root()
                .join("reviewerless_round")
                .to_string_lossy(),
            "reviewerless_ws",
        )
        .await;
        let board = crate::pipeline::board::store();
        let id =
            crate::util::test::make_ticket(board, &ws, "A ticket", TicketPhase::Verification).await;
        let ticket = crate::util::test::expect_ticket(board, &id).await;
        let job_id = &format!("reviewerless_job_{}", crate::generate_suffix());
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            job_id,
            "verification",
            "reviewer",
            &ws.name,
        )
        .timestamps(crate::db::now())
        .insert()
        .await
        .expect("seed the round's job row");
        let profile = Profile::resolve(Some(r#"{"stages":{"verification":["tester"]}}"#))
            .expect("a tester-only profile");

        let plan = RoundPlan::fresh(&ticket, &ws, job_id, None, &profile)
            .await
            .expect("the round is planned");

        assert!(plan.reviewers.is_empty(), "the profile staffs no reviewers");
        assert_eq!(plan.testers.len(), TESTER_COUNT, "the tester still runs");
    }
}
