use super::*;
use crate::board::TicketComment;
use crate::prompt::load_prompt;
use crate::util::test::make_ticket;
use crate::util::test::{
    JobRowBuilder, create_test_workspace, expect_ticket, expect_ticket_phase,
    init_management_test_stores,
};
use crate::workspace::test_ws_named;

#[test]
fn bounce_breaker_max_is_ten() {
    assert_eq!(
        crate::joint_verdict::MAX_BOUNCES,
        10,
        "MAX_BOUNCES must be 10"
    );
}

// ── transition_ticket_to_done — conditional notification ─────────

/// Shorthand for [`init_management_test_stores`] + [`create_test_workspace`]
/// with a generated `ws_{suffix}` / `/tmp/test_{suffix}` name/path.
///
/// Creates a **DB-backed** workspace (inserted into the test DB), unlike
/// [`setup_ticket`] which returns an in-memory workspace.
///
/// Each test must pass a unique `suffix` to avoid UNIQUE constraint
/// and cross-test pollution on the shared ticket buffer.
async fn setup_db_workspace(suffix: &str) -> crate::Workspace {
    init_management_test_stores().await;

    let ws_name = format!("ws_{suffix}");
    let ws_path = format!("/tmp/test_{suffix}");
    create_test_workspace(&ws_path, &ws_name).await
}

/// Shorthand for [`init_management_test_stores`] + [`test_ws_named`] +
/// [`TicketBuilder`].
///
/// Creates an in-memory workspace (no DB insertion) with the given `path`
/// and `name`, creates a ticket with `title` and starting `phase`, and
/// returns `(workspace, ticket_id)`.
async fn setup_ticket(
    ws_path: &str,
    ws_name: &str,
    title: &str,
    phase: TicketPhase,
) -> (crate::Workspace, String) {
    init_management_test_stores().await;
    let ws = test_ws_named(ws_path, ws_name);
    let ticket_id = make_ticket(board(), &ws, title, phase).await;
    (ws, ticket_id)
}

/// Verify the Buffer → Notify + drain sequence across two QaPassed tickets
/// via `transition_ticket_to_done`: the first one buffers, the last one
/// notifies and drains the buffer.
///
/// Serialized with the reset_inflight_tickets tests (shared global board)
/// and holds retry_tests_lock: the Notify path routes a Manager notification
/// whose consumer loop runs a Manager agent that reads the process-global
/// provider (project convention: retry_tests_lock).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn transition_ticket_to_done_buffer_and_notify() {
    let _lock = crate::util::test::retry_tests_lock();
    let ws = setup_db_workspace("drains_buffer").await;

    // Two QaPassed tickets in the same workspace
    let first_id = make_ticket(board(), &ws, "Ticket A", TicketPhase::QaPassed).await;
    let second_id = make_ticket(board(), &ws, "Ticket B", TicketPhase::QaPassed).await;

    let ticket_a = expect_ticket(board(), &first_id).await;

    // Transition ticket A — ticket B is still QaPassed (active), so Buffer
    transition_ticket_to_done(
        &ticket_a,
        TicketPhase::QaPassed,
        "Test — ticket A done, B still active",
    )
    .await;

    // Intermediate assertion: verify the Buffer path was actually taken.
    // Without this, a bug where has_active_tickets_excluding incorrectly
    // returns false (causing Notify instead of Buffer) would only be caught
    // by the final empty-buffer check — which could still pass if the Notify
    // path also happened to drain the buffer cleanly (e.g., by sending an
    // empty notification). Draining here verifies entry was pushed.
    let intermediate = crate::ticket_buffer::drain("ws_drains_buffer");
    assert!(
        !intermediate.is_empty(),
        "After first QaPassed → Done with other active tickets: \
             should have buffered the notification (got empty buffer)",
    );

    // Transition ticket B — no more active tickets, should Notify and drain
    let ticket_b = expect_ticket(board(), &second_id).await;
    transition_ticket_to_done(
        &ticket_b,
        TicketPhase::QaPassed,
        "Test — ticket B done, last ticket",
    )
    .await;

    // Verify both tickets are Done and have SYSTEM_ROLE comments
    for (id, label) in [(&first_id, "A"), (&second_id, "B")] {
        let t = expect_ticket(board(), id).await;
        assert_eq!(t.phase, TicketPhase::Done, "Ticket {label} should be Done");

        // Each Done transition should have written a SYSTEM_ROLE comment
        let comments = board().get_comments(id).await.expect("get_comments");
        assert!(
            comments.iter().any(|c| c.role == SYSTEM_ROLE),
            "Ticket {label}: expected SYSTEM_ROLE comment from transition_ticket_to_done"
        );
    }

    // No entries should remain for this workspace (the Notify path on
    // ticket B calls drain() internally; we drained the intermediate
    // buffer above, so this check is for leftover / stale entries).
    let drained = crate::ticket_buffer::drain("ws_drains_buffer");
    assert!(
        drained.is_empty(),
        "Buffer should be empty after last ticket's Notify drains it",
    );
}

// ── Setup helpers ──────────────────────────────────────────────────────

/// Shared helper: create a passing verdict (score >= REVIEW_QA_THRESHOLD).
fn pass_verdict() -> crate::Verdict {
    crate::Verdict {
        score: REVIEW_QA_THRESHOLD,
        issues_detected: vec![],
    }
}

/// Shared helper: create a failing verdict (score < REVIEW_QA_THRESHOLD).
fn fail_verdict() -> crate::Verdict {
    crate::Verdict {
        score: 3,
        issues_detected: vec!["No timeout check".into()],
    }
}

/// Helper: a `ParallelVerdict` with no response.
fn no_verdict() -> ParallelVerdict {
    ParallelVerdict::NoResponse("agent produced no response".into())
}

/// Helper: wrap a passing verdict (reviewer/QA flow).
fn pass_result() -> ParallelVerdict {
    ParallelVerdict::Verdict(pass_verdict())
}

/// Helper: wrap a failing verdict (reviewer/QA flow).
fn fail_result() -> ParallelVerdict {
    ParallelVerdict::Verdict(fail_verdict())
}

/// Helper: construct an analyst verdict with explicit score / issues.
fn analyst_verdict(score: u8, issues: &[&str]) -> ParallelVerdict {
    ParallelVerdict::Verdict(crate::Verdict {
        score,
        issues_detected: issues.iter().map(|&s| s.into()).collect(),
    })
}

/// Install the process-global test seams needed by the joint-verdict
/// synthesis path: a tiny retry policy (fast synthesis exhaustion) and a
/// scripted fake provider. Returns the guard RAII handles.
///
/// Without a fake provider, `providers::chat_scoped` panics on the unset
/// global — every test that drives an any-failed / partial round (which
/// triggers synthesis) must install these.
fn install_synthesis_test_seams(
    fake: crate::util::test::FakeProvider,
) -> (
    std::sync::MutexGuard<'static, ()>,
    crate::util::test::RetryPolicyGuard,
    crate::util::test::FakeProviderGuard,
) {
    let lock = crate::util::test::retry_tests_lock();
    let policy_guard =
        crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());
    let provider_guard = crate::util::test::install_fake_provider(std::sync::Arc::new(fake));
    (lock, policy_guard, provider_guard)
}

// ── process_verifier_verdicts — verdict processing ─────────────────────

/// Verify all verdict-processing outcomes:
/// - All failed → Failed (SYSTEM_ROLE failure comment only — no joint comment)
/// - Any failed → bounce-back to ReadyForDevelopment with pipeline
///   reservation, a single joint comment (role = stage name), and a bumped
///   bounce counter
/// - The 11th bounce → Failed (bounce circuit breaker)
/// - All passed (Reviewer) → Reviewed with a joint comment
/// - All passed (QA) → QaPassed with a joint comment
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn process_verifier_verdicts_cases() {
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        title: &'static str,
        phase: TicketPhase,
        results: Vec<ParallelVerdict>,
        vi: VerifierInfo,
        expected_phase: TicketPhase,
        expected_pipeline_reservation: bool,
        /// Expected bounce_count after processing (0 for non-bounce cases).
        expected_bounce_count: i64,
        /// Expected number of stage-role (joint) comments after the round.
        expected_joint_comments: usize,
    }

    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let cases = vec![
        Case {
            name: "all failed -> Failed",
            ws_suffix: "vp_all_fail",
            title: "VP All Failed",
            phase: TicketPhase::InReview,
            results: vec![no_verdict(); 3],
            vi: REVIEWER_VI,
            expected_phase: TicketPhase::Failed,
            expected_pipeline_reservation: false,
            expected_bounce_count: 0,
            expected_joint_comments: 0,
        },
        Case {
            name: "any failed -> bounce-back with pipeline reservation",
            ws_suffix: "vp_any_fail",
            title: "VP Any Failed",
            phase: TicketPhase::InReview,
            results: vec![pass_result(), fail_result(), pass_result()],
            vi: REVIEWER_VI,
            expected_phase: TicketPhase::ReadyForDevelopment,
            expected_pipeline_reservation: true,
            expected_bounce_count: 1,
            expected_joint_comments: 1,
        },
        Case {
            name: "all passed -> Reviewed",
            ws_suffix: "vp_all_pass",
            title: "VP All Pass",
            phase: TicketPhase::InReview,
            results: vec![pass_result(), pass_result(), pass_result()],
            vi: REVIEWER_VI,
            expected_phase: TicketPhase::Reviewed,
            expected_pipeline_reservation: false,
            expected_bounce_count: 0,
            expected_joint_comments: 1,
        },
        Case {
            name: "all passed (QA) -> QaPassed",
            ws_suffix: "vp_qa_pass",
            title: "VP QA Pass",
            phase: TicketPhase::InQa,
            results: vec![pass_result(), pass_result(), pass_result()],
            vi: QA_VI,
            expected_phase: TicketPhase::QaPassed,
            expected_pipeline_reservation: false,
            expected_bounce_count: 0,
            expected_joint_comments: 1,
        },
    ];

    for case in &cases {
        let ws = test_ws_named("/tmp/test", case.ws_suffix);
        let ticket_id = make_ticket(board(), &ws, case.title, case.phase).await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        process_verifier_verdicts(&ws, &ticket, &case.results, case.vi).await;

        let ticket = expect_ticket(board(), &ticket_id).await;
        assert_eq!(
            ticket.phase, case.expected_phase,
            "case {}: expected phase {:?}, got {:?}",
            case.name, case.expected_phase, ticket.phase,
        );
        assert_eq!(
            ticket.pipeline_reservation, case.expected_pipeline_reservation,
            "case {}: expected pipeline_reservation={}, got {}",
            case.name, case.expected_pipeline_reservation, ticket.pipeline_reservation,
        );
        assert_eq!(
            ticket.bounce_count, case.expected_bounce_count,
            "case {}: expected bounce_count={}, got {}",
            case.name, case.expected_bounce_count, ticket.bounce_count,
        );

        // Every non-all-failed round writes exactly ONE joint comment (role =
        // stage name), replacing the three per-agent comments; all-failed
        // rounds keep only the SYSTEM_ROLE failure comment.
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get comments");
        let verdict_comments: Vec<&TicketComment> = comments
            .iter()
            .filter(|c| c.role == stage_name(case.vi.role))
            .collect();
        if case.expected_joint_comments == 1 {
            assert_eq!(
                verdict_comments.len(),
                1,
                "case {}: one joint comment expected, got {}",
                case.name,
                verdict_comments.len(),
            );
            let joint = &verdict_comments[0];
            assert!(
                !joint.content.contains("valid verdicts")
                    && !joint.content.contains("threshold 9/10"),
                "case {}: verifier comments carry no round headline or threshold line: {}",
                case.name,
                joint.content,
            );
            assert!(
                joint.content.contains("### Summary"),
                "case {}: joint comment keeps the Summary section: {}",
                case.name,
                joint.content,
            );
        } else {
            assert!(
                verdict_comments.is_empty(),
                "case {}: no per-stage comments expected for all-failed rounds",
                case.name,
            );
        }
    }
}

/// The stage-finalization choke point treats a phase-guard miss (ticket moved
/// externally while the stage was finishing) as a first-class, expected
/// outcome: the whole round is rolled back silently — nothing is written, no
/// bounce is counted — and `assigned_to` is NOT cleared by the finalizer (the
/// external mover already handled the ticket; a finalizer clear would clobber
/// a fresh claim by the new phase).
///
/// The round uses a mixed verdict (pass/fail/pass → any_failed), so the
/// target is ReadyForDevelopment and the closure increments the bounce
/// counter: the `bounce_count == 0` assertion genuinely exercises the
/// rollback of an in-transaction write that WOULD have committed had the
/// guard applied.
///
/// Regression guard for the structural fix: the guard miss is a silent skip
/// (the silent-rollback path never reaches the warn arms), not an error —
/// this test asserts the observable side effects (phase untouched, bounce
/// unchanged, assignment preserved, no comment written).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn verifier_finalization_on_moved_ticket_is_clean_skip() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = test_ws_named("/tmp/test", "vp_moved");
    let ticket_id = make_ticket(board(), &ws, "VP Moved", TicketPhase::InReview).await;

    // The Manager triages the ticket to Planning while the verifier round
    // finishes. The mover's claim on the ticket must survive the finalizer.
    board()
        .transition_to(&ticket_id, None, TicketPhase::Planning, None)
        .await
        .expect("external move to Planning");
    board()
        .set_assigned_to_no_cancel(&ticket_id, Some("external-mover"))
        .await
        .expect("external mover claims the ticket");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let transitioned = process_verifier_verdicts(
        &ws,
        &ticket,
        // Mixed verdict → any_failed → target ReadyForDevelopment: the
        // closure writes a joint comment AND increments the bounce counter,
        // so the guard miss must roll both back.
        &[pass_result(), fail_result(), pass_result()],
        REVIEWER_VI,
    )
    .await;
    assert!(!transitioned, "guard miss must not report a transition");

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::Planning,
        "the moved ticket must be left untouched"
    );
    assert_eq!(
        ticket.bounce_count, 0,
        "guard miss must not bump the bounce counter"
    );
    assert_eq!(
        ticket.assigned_to.as_deref(),
        Some("external-mover"),
        "guard miss must not clear the external mover's assignment"
    );

    // Nothing was written: a skipped round leaves no joint comment behind.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    assert!(
        !comments
            .iter()
            .any(|c| c.role == stage_name(REVIEWER_VI.role)),
        "a skipped round must not write a joint comment"
    );
}

/// Resume a review round from stored roster outcomes: done slots replay their
/// checkpointed verdicts (no re-invocation, no LLM for the agents) and the
/// verdicts are re-processed through the existing process_verifier_verdicts —
/// the ticket transitions to Reviewed exactly like a fresh round, with no
/// double bounce. High-signal replay test: the round-1 analyze-resume bugs lived
/// in this path (job-id conflicts, caller misrouting).
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the InReview fixture).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn resume_verifier_round_replays_stored_outcomes() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = test_ws_named("/tmp/test", "vp_resume");
    let ticket_id = make_ticket(board(), &ws, "VP Resume", TicketPhase::InReview).await;
    let job_id = "vp_resume_job";
    let now = crate::turso::now();
    let conn = &crate::session::store().conn;

    // Job row + ticket_stage_jobs row exactly as a crashed dispatch leaves
    // them (stage=review, phase=in_review, round=1).
    JobRowBuilder::new(conn, job_id, "ticket_stage", "reviewer", &ws.name)
        .timestamps(now.clone())
        .insert()
        .await
        .unwrap();
    conn.execute(
        "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
         VALUES (?1, ?2, 'review', 'in_review', 1)",
        crate::turso::params![job_id, ticket_id.clone()],
    )
    .await
    .unwrap();

    // Three done roster rows carrying stored passing verdicts (the replay
    // reconstructs these WITHOUT calling the provider — the FakeProvider is
    // only needed for the joint-comment synthesis).
    for i in 0..3 {
        let agent_id = format!("ticket_{ticket_id}_resume_{i}_reviewer");
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
             VALUES (?1, ?2, 'verifier', ?3, 'done', ?4, '')",
            crate::turso::params![
                job_id,
                agent_id,
                i64::from(i),
                serialize_verdict_outcome(&pass_result()),
            ],
        )
        .await
        .unwrap();
    }

    let ticket = expect_ticket(board(), &ticket_id).await;
    resume_ticket_stage_round("review".to_string(), job_id.to_string(), ticket, ws).await;

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::Reviewed,
        "replayed all-pass stored outcomes must transition to Reviewed"
    );
    // No double bounce on replay (bounce_count only increments on failures).
    assert_eq!(ticket.bounce_count, 0, "no bounce on an all-pass replay");
    // Job completed (status=done, roster cascaded).
    let jobs = conn
        .query(
            "SELECT status FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].get::<String>(0).unwrap(), "done");
    let agents = conn
        .query(
            "SELECT COUNT(*) FROM agents WHERE job_id = ?1",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
    assert_eq!(
        agents[0].get::<i64>(0).unwrap(),
        0,
        "roster rows cascaded on completion"
    );
}

/// The phase-gate bail path returns before run_agent runs, so its exit
/// guard never fires and the router entry would leak a dead sender; the
/// explicit unregister in the bail path is the only cleanup. Pins the
/// cleanup via router_contains (try_route cannot distinguish an absent
/// entry from a dead receiver).
///
/// Serialized with reset_inflight (shared global board — the stale-Phase
/// fixture transitions the ticket out of Analysis).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn parallel_round_phase_gate_bail_unregisters_router() {
    init_management_test_stores().await;
    let ws = test_ws_named("/tmp/test", "pg_bail");
    let ticket_id = make_ticket(board(), &ws, "PG Bail", TicketPhase::Analysis).await;
    // Stale Arc: phase=Analysis loaded BEFORE the DB row moves to Planning —
    // the member gate compares the Arc's phase against the live DB.
    let ticket = Arc::new(expect_ticket(board(), &ticket_id).await);
    assert_eq!(ticket.phase, TicketPhase::Analysis);
    board()
        .transition_to(&ticket_id, None, TicketPhase::Planning, None)
        .await
        .unwrap();
    let slots: Vec<TicketStageSlot> = (0..2)
        .map(|i| TicketStageSlot {
            idx: i,
            agent_id: format!("pg_bail_{i}"),
            task: "task".to_string(),
            status: crate::jobs::RowStatus::Launched,
            outcome: None,
        })
        .collect();
    let results = run_parallel_agents(
        &ticket,
        &ws,
        Role::Analyst,
        "extract",
        "pg_bail_job",
        &slots,
        false,
    )
    .await;
    assert_eq!(results.len(), 2);
    for (slot, result) in slots.iter().zip(&results) {
        assert!(
            matches!(result, ParallelVerdict::NoResponse(r) if r == PHASE_GATE_BAIL_REASON),
            "member must bail at the phase gate with the neutral reason"
        );
        assert!(
            !crate::message_router::router_contains(&slot.agent_id),
            "phase-gate bail must unregister the router entry for {}",
            slot.agent_id,
        );
    }
}

/// A panicked round member maps to a contained NoResponse (fail-open) with
/// the scrubbed panic message as the reason — the round continues, matching
/// the analyze/research precedent.
#[tokio::test]
async fn panicked_round_member_maps_to_contained_no_response() {
    let handle = tokio::spawn(async {
        panic!("probe api_key=sk-abcdefgh12345678 boom");
    });
    let verdict = round_member_failed(handle.await.unwrap_err());
    let ParallelVerdict::NoResponse(reason) = verdict else {
        panic!("panicked member must resolve to NoResponse");
    };
    assert!(reason.contains("probe") && reason.contains("boom"));
    assert!(
        !reason.contains("sk-abcdefgh12345678"),
        "panic reason must be credential-scrubbed"
    );
}

/// Resume an engineer round at boot (S5): the NULL-seat anchor session is
/// CONTINUED — the resume dispatches with an empty message (session content
/// exists → no duplicate task-prompt append), the engineer completes, the
/// ticket transitions to InDiagnostics, the job completes, the roster
/// cascades, and the anchor row itself survives (permanent seat).
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the InDevelopment fixture).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn resume_engineer_round_continues_anchor_session() {
    init_management_test_stores().await;
    // Main LLM response + engineer-summary extraction (2 calls).
    let fake = crate::util::test::FakeProvider::new()
        .ok("implemented the resume path")
        .ok(r#"{"items": ["resumed the engineer session"]}"#);
    let (_lock, _policy_guard, _provider_guard) = install_synthesis_test_seams(fake);

    let ws = test_ws_named("/tmp/test", "eng_resume");
    let ticket_id = make_ticket(board(), &ws, "Eng Resume", TicketPhase::InDevelopment).await;
    let job_id = "eng_resume_job";
    let now = crate::turso::now();
    let conn = &crate::session::store().conn;
    let anchor_id = crate::jobs::engineer_anchor_id(&ticket_id);
    let task = "round 2 feedback: fix the tests";

    // Job + ticket_stage_jobs rows exactly as a crashed round-2 dispatch
    // leaves them (stage=engineer, phase=in_development, round=2).
    JobRowBuilder::new(conn, job_id, "ticket_stage", "engineer", &ws.name)
        .task(task)
        .timestamps(now.clone())
        .insert()
        .await
        .unwrap();
    conn.execute(
        "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
         VALUES (?1, ?2, 'engineer', 'in_development', 2)",
        crate::turso::params![job_id, ticket_id.clone()],
    )
    .await
    .unwrap();
    // NULL-seat anchor + this round's roster row (same agent_id, coexisting
    // under the composite PK).
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
         VALUES (NULL, ?1, 'engineer', NULL, 'done', ?2)",
        crate::turso::params![anchor_id.clone(), task],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
         VALUES (?1, ?2, 'engineer', 0, 'launched', ?3)",
        crate::turso::params![job_id, anchor_id.clone(), task],
    )
    .await
    .unwrap();
    // Round-1 session content exists → empty-message dispatch (no duplicate
    // task-prompt append).
    conn.execute(
        "INSERT INTO sessions (agent_id, role, content, created_at) \
         VALUES (?1, 'user', ?2, ?3)",
        crate::turso::params![
            anchor_id.clone(),
            format!("<ts>{now}</ts>\n\nround 1 task"),
            now.clone()
        ],
    )
    .await
    .unwrap();

    let ticket = expect_ticket(board(), &ticket_id).await;
    resume_ticket_stage_round("engineer".to_string(), job_id.to_string(), ticket, ws).await;

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::InDiagnostics,
        "resumed engineer must transition to InDiagnostics"
    );
    // Job completed + roster cascaded; the anchor survives.
    let jobs = conn
        .query(
            "SELECT status FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].get::<String>(0).unwrap(), "done");
    let roster = conn
        .query(
            "SELECT COUNT(*) FROM agents WHERE job_id = ?1",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
    assert_eq!(
        roster[0].get::<i64>(0).unwrap(),
        0,
        "roster cascaded on completion"
    );
    let anchors = conn
        .query(
            "SELECT COUNT(*) FROM agents WHERE agent_id = ?1 AND job_id IS NULL",
            crate::turso::params![anchor_id.clone()],
        )
        .await
        .unwrap();
    assert_eq!(
        anchors[0].get::<i64>(0).unwrap(),
        1,
        "NULL-seat anchor survives round completion"
    );
    // Session continuity: exactly the round-1 user row + the round-2
    // assistant response — the task was NOT re-appended.
    let msgs = conn
        .query(
            "SELECT role, content FROM sessions WHERE agent_id = ?1 ORDER BY id",
            crate::turso::params![anchor_id.clone()],
        )
        .await
        .unwrap();
    assert_eq!(
        msgs.len(),
        2,
        "round-1 user row + round-2 assistant response only"
    );
    assert_eq!(msgs[0].get::<String>(0).unwrap(), "user");
    assert_eq!(msgs[1].get::<String>(0).unwrap(), "assistant");
    assert!(
        !msgs[1]
            .get::<String>(1)
            .unwrap()
            .contains("round 2 feedback"),
        "task must not be re-appended on a resumed session"
    );
}

/// Resume a sanitation round at boot: the job-derived session
/// `ticket_{job_id}_sanitation` CONTINUES (empty message — no duplicate
/// task-prompt append), the verdict is extracted from the resumed response and
/// processed, the ticket transitions to SanitationPassed, the job completes,
/// and the roster cascades.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the InSanitation fixture).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn resume_sanitation_round_continues_session_and_passes() {
    init_management_test_stores().await;
    // Main LLM response + SanitationVerdict extraction (2 calls).
    let fake = crate::util::test::FakeProvider::new()
        .ok("inspected the workspace — no garbage files found")
        .ok(r#"{"pass": true, "garbage_files": [], "rationale": "workspace is clean"}"#);
    let (_lock, _policy_guard, _provider_guard) = install_synthesis_test_seams(fake);

    let ws = test_ws_named("/tmp/test", "san_resume");
    let ticket_id = make_ticket(board(), &ws, "San Resume", TicketPhase::InSanitation).await;
    let job_id = "san_resume_job";
    let now = crate::turso::now();
    let conn = &crate::session::store().conn;
    let agent_id = format!("ticket_{job_id}_sanitation");
    let task = "sanitation task";

    // Job + ticket_stage_jobs rows exactly as a crashed dispatch leaves them
    // (stage=sanitation, phase=in_sanitation, round=1).
    JobRowBuilder::new(conn, job_id, "ticket_stage", "sanitation", &ws.name)
        .task(task)
        .timestamps(now.clone())
        .insert()
        .await
        .unwrap();
    conn.execute(
        "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
         VALUES (?1, ?2, 'sanitation', 'in_sanitation', 1)",
        crate::turso::params![job_id, ticket_id.clone()],
    )
    .await
    .unwrap();
    // Roster row + session content → empty-message dispatch (no re-append).
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
         VALUES (?1, ?2, 'sanitation', 0, 'launched', ?3)",
        crate::turso::params![job_id, agent_id.clone(), task],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO sessions (agent_id, role, content, created_at) \
         VALUES (?1, 'user', ?2, ?3)",
        crate::turso::params![
            agent_id.clone(),
            format!("<ts>{now}</ts>\n\n{task}"),
            now.clone()
        ],
    )
    .await
    .unwrap();

    let ticket = expect_ticket(board(), &ticket_id).await;
    resume_ticket_stage_round("sanitation".to_string(), job_id.to_string(), ticket, ws).await;

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::SanitationPassed,
        "resumed sanitation pass must transition to SanitationPassed"
    );
    // Job completed + roster cascaded.
    let jobs = conn
        .query(
            "SELECT status FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].get::<String>(0).unwrap(), "done");
    let roster = conn
        .query(
            "SELECT COUNT(*) FROM agents WHERE job_id = ?1",
            crate::turso::params![job_id],
        )
        .await
        .unwrap();
    assert_eq!(
        roster[0].get::<i64>(0).unwrap(),
        0,
        "roster cascaded on completion"
    );
    // Session continuity: exactly the seeded user row + the resumed assistant
    // response — the task was NOT re-appended.
    let msgs = conn
        .query(
            "SELECT role, content FROM sessions WHERE agent_id = ?1 ORDER BY id",
            crate::turso::params![agent_id.clone()],
        )
        .await
        .unwrap();
    assert_eq!(msgs.len(), 2, "seeded user row + assistant response only");
    assert_eq!(msgs[0].get::<String>(0).unwrap(), "user");
    assert_eq!(msgs[1].get::<String>(0).unwrap(), "assistant");
    assert!(
        !msgs[1].get::<String>(1).unwrap().contains(task),
        "task must not be re-appended on a resumed session"
    );
}

/// Provider that panics on the first scoped call. `drain_first` starts the
/// process-global graceful drain right before panicking — simulating a drain
/// that begins while the agent is mid-LLM-call (the dispatch task's panic
/// recovery then observes `aborting() == true`).
struct PanicProvider {
    drain_first: bool,
}

#[async_trait::async_trait]
impl crate::Provider for PanicProvider {
    async fn chat_scoped(
        &self,
        _request: crate::ChatRequest,
        _idle_timeout: std::time::Duration,
        _deadline: std::time::Instant,
    ) -> Result<crate::ChatResponse, crate::providers::ScopedCallError> {
        if self.drain_first {
            crate::shutdown::drain_begin();
        }
        panic!("provider boom");
    }
}

/// The dispatch-panic drain guard: a panic in the dispatch task while the
/// graceful drain is active must NOT drive the exit-time ticket rollback (no
/// Failed transition, no failure comment) — the job stays status='launched' for
/// boot resume. The control case (no drain) proves the same panic path DOES
/// transition to Failed when the guard is inactive.
///
/// Serialized with the reset_inflight_tickets tests (shared global board + the
/// process-global drain flag).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn dispatch_panic_during_drain_skips_failed_transition() {
    init_management_test_stores().await;
    let _lock = crate::util::test::retry_tests_lock();
    let _policy_guard =
        crate::util::test::install_test_retry_policy(crate::retry::tiny_test_policy());

    // ── Control: panic WITHOUT drain → Failed transition ──────────────
    let ctrl_provider =
        crate::util::test::install_fake_provider(std::sync::Arc::new(PanicProvider {
            drain_first: false,
        }));
    let ws_ctrl = test_ws_named("/tmp/test", "panic_ctrl");
    let ctrl_id = make_ticket(
        board(),
        &ws_ctrl,
        "Panic Control",
        TicketPhase::InDevelopment,
    )
    .await;
    let ticket = expect_ticket(board(), &ctrl_id).await;
    spawn_dispatch(PollPhase::EngineerDevelopment, ticket, ws_ctrl);
    // The dispatch task is fire-and-forget — poll for the Failed transition.
    let mut reached_failed = false;
    for _ in 0..50 {
        if expect_ticket_phase(board(), &ctrl_id).await == TicketPhase::Failed {
            reached_failed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        reached_failed,
        "control panic must transition the ticket to Failed"
    );
    drop(ctrl_provider);

    // ── Drain case: panic while the drain begins → NO Failed transition ──
    let drain_provider =
        crate::util::test::install_fake_provider(std::sync::Arc::new(PanicProvider {
            drain_first: true,
        }));
    let ws_drain = test_ws_named("/tmp/test", "panic_drain");
    let drain_id = make_ticket(
        board(),
        &ws_drain,
        "Panic Drain",
        TicketPhase::InDevelopment,
    )
    .await;
    let ticket = expect_ticket(board(), &drain_id).await;
    spawn_dispatch(PollPhase::EngineerDevelopment, ticket, ws_drain);
    // Wait for the provider to fire (the drain flag flips synchronously with
    // the panic; the unwind + guard are synchronous from there).
    let mut fired = false;
    for _ in 0..50 {
        if crate::shutdown::is_draining() {
            fired = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(fired, "dispatch task never reached the provider");
    // Clear the process-global drain flag BEFORE the assertions so a failing
    // assert cannot poison the serialized group.
    crate::shutdown::drain_clear();
    drop(drain_provider);

    let t = expect_ticket(board(), &drain_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::InDevelopment,
        "a panic during the drain must NOT transition the ticket to Failed"
    );
    let comments = board().get_comments(&drain_id).await.unwrap();
    assert!(
        !comments
            .iter()
            .any(|c| c.content.contains("Dispatch panicked")),
        "no failure comment during the drain (job resumes at boot)"
    );
}

/// The bounce circuit breaker: a ticket already at [`MAX_BOUNCES`] bounces
/// fails on the next failed round instead of bouncing again.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn eleventh_bounce_fails_ticket() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = test_ws_named("/tmp/test", "eleventh_bounce");
    let ticket_id = make_ticket(board(), &ws, "Eleventh Bounce", TicketPhase::InReview).await;
    board()
        .conn
        .execute(
            "UPDATE tickets SET bounce_count = ?1 WHERE id = ?2",
            turso::params![
                i64::try_from(crate::joint_verdict::MAX_BOUNCES).unwrap(),
                ticket_id.as_str()
            ],
        )
        .await
        .expect("set bounce_count to max");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let transitioned = process_verifier_verdicts(
        &ws,
        &ticket,
        &[pass_result(), fail_result(), pass_result()],
        REVIEWER_VI,
    )
    .await;
    assert!(
        transitioned,
        "11th-bounce round should transition the ticket"
    );

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::Failed,
        "11th bounce must fail the ticket"
    );
    assert_eq!(
        ticket.bounce_count,
        i64::try_from(crate::joint_verdict::MAX_BOUNCES).unwrap(),
        "bounce counter stays at the max — the failing bounce is not counted"
    );
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    assert!(
        comments
            .iter()
            .any(|c| c.content.contains("circuit breaker")),
        "the bounce breaker must leave an explicit trip comment",
    );
}

/// Regression guard for the auto-pause trigger boundaries:
///
/// A technical failure (all verifier agents failing) pauses the workspace.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases). Also serialized
/// with the drain-flag writers: `process_verifier_verdicts` and
/// `pause_workspace_on_failure` consult the process-global drain flag, which
/// would suppress the pause and the transition (project convention:
/// retry_tests_lock).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn technical_failure_pauses_workspace() {
    let _lock = crate::util::test::retry_tests_lock();
    init_management_test_stores().await;

    let ws_verifier = create_test_workspace("/tmp/pause_verifier_ws", "ws_pause_verifier").await;
    let verifier_id = make_ticket(
        board(),
        &ws_verifier,
        "Verifier All Failed",
        TicketPhase::InReview,
    )
    .await;
    let ticket = expect_ticket(board(), &verifier_id).await;
    let transitioned =
        process_verifier_verdicts(&ws_verifier, &ticket, &vec![no_verdict(); 3], REVIEWER_VI).await;
    assert!(
        transitioned,
        "all-failed round should transition the ticket"
    );
    let ws = crate::workspace::store()
        .get_by_name("ws_pause_verifier")
        .await
        .expect("get workspace")
        .expect("workspace exists");
    assert!(ws.paused, "verifier all-failed must pause the workspace");

    // The failure comment must carry the pause notice + a per-agent reason.
    let comments = board()
        .get_comments(&verifier_id)
        .await
        .expect("get comments");
    let last = comments
        .last()
        .expect("failure comment written")
        .content
        .clone();
    assert!(last.contains("Workspace paused"), "{last}");
    assert!(last.contains("agent produced no response"), "{last}");
}

// ── Claim gate: automatic claims blocked while paused or not Ready ──

/// The claim gate ([`blocks_claim`]) must block exactly the automatic pickup
/// of *new* work: BacklogAnalysis (backlog → analysis) and
/// EngineerDevelopment (ready_for_development → in_development). Two gates:
///
/// * **Pause gate** — later-phase claims (review, QA) proceed while paused,
///   and nothing is blocked when a Ready workspace is unpaused.
/// * **Status gate** — a non-Ready workspace (Pending/Analyzing/Failed)
///   blocks new-work claims even when unpaused: its contexts are missing or
///   stale, so a manual unpause must not re-enable development work.
///
/// Asserted against [`CLAIM_PHASES`] — the only phases the gate predicate is
/// ever consulted for in production — so the test cannot drift from the
/// claim pipeline's real surface (SanitationCheck/DiagnosticsCheck never
/// flow through the claim loop and are deliberately absent).
#[test]
fn blocks_claim_gate_matrix() {
    for &(source, phase) in CLAIM_PHASES {
        let new_work = matches!(
            phase,
            PollPhase::BacklogAnalysis | PollPhase::EngineerDevelopment
        );
        // Pause gate: a paused Ready workspace blocks new work only.
        let paused = Workspace {
            status: WorkspaceStatus::Ready,
            paused: true,
            ..Default::default()
        };
        assert_eq!(
            blocks_claim(&paused, phase),
            new_work,
            "pause gate mismatch for {} ({} → {})",
            phase.info().log_label,
            source.as_ref(),
            phase.info().expected_phase.as_ref(),
        );
        // Baseline: a Ready + unpaused workspace runs everything.
        let ready = Workspace {
            status: WorkspaceStatus::Ready,
            paused: false,
            ..Default::default()
        };
        assert!(
            !blocks_claim(&ready, phase),
            "{} must run when Ready and unpaused",
            phase.info().log_label,
        );
        // Status gate: non-Ready + unpaused still blocks new work (missing or
        // stale contexts) but lets later phases through.
        for status in [
            WorkspaceStatus::Pending,
            WorkspaceStatus::Analyzing,
            WorkspaceStatus::Failed,
        ] {
            let not_ready = Workspace {
                status,
                paused: false,
                ..Default::default()
            };
            assert_eq!(
                blocks_claim(&not_ready, phase),
                new_work,
                "status gate mismatch for {} ({})",
                phase.info().log_label,
                status,
            );
        }
    }
}

/// Regression test for the pause gate: a paused workspace must keep its
/// Backlog and ReadyForDevelopment tickets unclaimed, and unpausing resumes
/// the automatic claims on the next pipeline run.
///
/// Later-phase claims (review/QA) are not part of the gate — covered by
/// [`blocks_claim_gate_matrix`]; this test exercises the real
/// [`run_claim_pipeline`] path for the two gated phases.
///
/// Serialized with the reset_inflight tests: `run_claim_pipeline` claims
/// through the shared global board. The unpaused run spawns real dispatch
/// tasks (they abort when this test's runtime drops); the claim transitions
/// are synchronous, so the phase assertions are deterministic.
///
/// Two deliberate couplings, both serialized by the group + lock:
/// - The spawned dispatch tasks read the process-global provider, so the
///   test holds `retry_tests_lock()` per the suite's provider-seam
///   convention.
/// - The spawned dispatches may persist `ticket_stage_jobs`/`agents` rows
///   for this test's ticket IDs into the shared test jobs DB before the
///   runtime drops them; later `recover_from_restart` tests in this serial
///   group tolerate extra rows (they assert on their own fixture IDs).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global provider seams
async fn paused_workspace_holds_backlog_and_rfd_until_unpause() {
    let _lock = crate::util::test::retry_tests_lock();
    let ws = setup_db_workspace("pause_gate").await;
    let ws_name = ws.name.clone();
    // A workspace with tickets has completed discovery — the claim gate's
    // status check requires Ready (Pending/Analyzing/Failed workspaces never
    // take new-work claims, even unpaused).
    crate::workspace::store()
        .set_status(&ws_name, &WorkspaceStatus::Ready)
        .await
        .expect("set workspace ready");
    crate::workspace::store()
        .set_paused(&ws_name, true)
        .await
        .expect("pause workspace");

    // Backlog claims carry a 5s fresh-ticket grace (BACKLOG_CLAIM_GRACE) —
    // backdate so the claim is eligible once unpaused.
    let backlog_id = make_ticket(board(), &ws, "Paused Backlog", TicketPhase::Backlog).await;
    let rfd_id = make_ticket(board(), &ws, "Paused RFD", TicketPhase::ReadyForDevelopment).await;
    let old = (chrono::Utc::now() - ChronoDuration::minutes(10)).to_rfc3339();
    for id in [&backlog_id, &rfd_id] {
        board()
            .conn
            .execute(
                "UPDATE tickets SET created_at = ?1 WHERE id = ?2",
                crate::turso::params![old.clone(), id.clone()],
            )
            .await
            .expect("backdate ticket created_at");
    }

    // Paused: neither automatic claim fires.
    let paused_ws = crate::workspace::store()
        .get_by_name(&ws_name)
        .await
        .expect("get workspace")
        .expect("workspace exists");
    run_claim_pipeline(&paused_ws).await;
    assert_eq!(
        expect_ticket_phase(board(), &backlog_id).await,
        TicketPhase::Backlog,
        "paused workspace must not claim backlog into analysis",
    );
    assert_eq!(
        expect_ticket_phase(board(), &rfd_id).await,
        TicketPhase::ReadyForDevelopment,
        "paused workspace must not claim RFD into development",
    );

    // Unpaused: the next pipeline run claims both automatically.
    crate::workspace::store()
        .set_paused(&ws_name, false)
        .await
        .expect("unpause workspace");
    let resumed_ws = crate::workspace::store()
        .get_by_name(&ws_name)
        .await
        .expect("get workspace")
        .expect("workspace exists");
    run_claim_pipeline(&resumed_ws).await;
    assert_eq!(
        expect_ticket_phase(board(), &backlog_id).await,
        TicketPhase::Analysis,
        "unpaused workspace must claim backlog into analysis",
    );
    assert_eq!(
        expect_ticket_phase(board(), &rfd_id).await,
        TicketPhase::InDevelopment,
        "unpaused workspace must claim RFD into development",
    );
}

// ── process_analyst_verdicts — analyst scoring and transitions ─────────

/// Verify process_analyst_verdicts across all outcomes (fail-open):
/// - All analysts pass → Planning with one joint comment (role "Analysis")
/// - Partial fail → Planning with one joint comment (role "Analysis")
/// - No verdicts → Planning with a joint comment carrying the failure dumps
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn process_analyst_verdicts_cases() {
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        title: &'static str,
        results: Vec<ParallelVerdict>,
        /// Substring that must appear in the joint comment.
        expected_comment_substring: &'static str,
    }

    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let cases = vec![
        Case {
            name: "all pass -> Planning with joint comment",
            ws_suffix: "an_all_pass",
            title: "Analyst All Pass",
            results: vec![
                analyst_verdict(10, &[]),
                analyst_verdict(9, &[]),
                analyst_verdict(8, &[]),
            ],
            expected_comment_substring: "All LGTM",
        },
        Case {
            name: "partial fail -> Planning with joint comment",
            ws_suffix: "an_partial",
            title: "Analyst Partial Fail",
            results: vec![
                analyst_verdict(10, &[]),
                analyst_verdict(3, &["Missing data"]),
                analyst_verdict(8, &["Minor issue"]),
            ],
            expected_comment_substring: "flagged potential blockers",
        },
        Case {
            name: "no verdicts -> Planning with failure dumps",
            ws_suffix: "an_no_v",
            title: "Analyst No Verdicts",
            results: vec![no_verdict(); 3],
            expected_comment_substring: "Agent failures",
        },
    ];

    for case in &cases {
        let ws = test_ws_named("/tmp/test", case.ws_suffix);
        let ticket_id = make_ticket(board(), &ws, case.title, TicketPhase::Analysis).await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        process_analyst_verdicts(&ws, &ticket, &case.results).await;

        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
            TicketPhase::Planning,
            "case {}: analysis is fail-open — expected Planning, got {:?}",
            case.name,
            phase,
        );

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        let verdict_comments: Vec<&TicketComment> = comments
            .iter()
            .filter(|c| c.role == stage_name(Role::Analyst))
            .collect();
        assert_eq!(
            verdict_comments.len(),
            1,
            "case {}: exactly one joint comment expected, got {}",
            case.name,
            verdict_comments.len(),
        );
        assert!(
            verdict_comments[0]
                .content
                .contains(case.expected_comment_substring),
            "case {}: joint comment should contain {:?}, got: {}",
            case.name,
            case.expected_comment_substring,
            verdict_comments[0].content,
        );
    }
}

/// Fail-open integration: a round where the joint-comment synthesis is
/// unavailable (provider scripted to fail) still advances the ticket with a
/// deterministic fallback comment — never a fabricated one.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn analyst_round_fails_open_with_fallback_comment() {
    init_management_test_stores().await;
    // Script every synthesis attempt as a transport failure → exhaustion →
    // deterministic fallback.
    let fake = crate::util::test::FakeProvider::new()
        .err(crate::retry::FailureClass::Transport, "synthesis down")
        .err(crate::retry::FailureClass::Transport, "synthesis down")
        .err(crate::retry::FailureClass::Transport, "synthesis down");
    let (_lock, _policy_guard, _provider_guard) = install_synthesis_test_seams(fake);

    let ws = test_ws_named("/tmp/test", "an_fail_open");
    let ticket_id = make_ticket(board(), &ws, "Fail Open", TicketPhase::Analysis).await;

    let results = vec![
        analyst_verdict(10, &[]),
        analyst_verdict(3, &["Missing data"]),
    ];
    let ticket = expect_ticket(board(), &ticket_id).await;
    process_analyst_verdicts(&ws, &ticket, &results).await;

    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(phase, TicketPhase::Planning, "fail-open must advance");

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    let joint = comments
        .iter()
        .find(|c| c.role == stage_name(Role::Analyst))
        .expect("joint comment written");
    assert!(
        joint.content.contains("LLM grouping unavailable"),
        "fallback marker must be explicit: {}",
        joint.content,
    );
    assert!(
        joint.content.contains("Missing data"),
        "deterministic issues must render: {}",
        joint.content,
    );
}

// ── handle_qa_passed — QA → Done path ───────────────────────────────

/// handle_qa_passed first checks whether git is available and whether the
/// workspace path is a git repo. In test environments git may exist, but
/// the workspace path is deliberately not a git repo, so the function
/// transitions directly to Done without committing.
/// This test validates the graceful non-git fallback path.
#[tokio::test]
async fn handle_qa_passed_no_git_to_done() {
    // Use a temporary directory without git init — guarantees no git repo
    // exists regardless of the test runner's filesystem state. The `dir`
    // binding must stay alive (function scope) for the workspace path to
    // remain valid.
    let dir = tempfile::tempdir().expect("create temp dir");
    let ws_path = dir.path().to_str().expect("temp path is valid UTF-8");
    let (ws, ticket_id) =
        setup_ticket(ws_path, "qa_no_git", "QA No Git", TicketPhase::QaPassed).await;

    let ticket = expect_ticket(board(), &ticket_id).await;

    handle_qa_passed(ticket, ws).await;

    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::Done,
        "QA passed should eventually transition to Done"
    );

    // Verify a SYSTEM_ROLE comment was written capturing the reason.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert!(
        comments
            .iter()
            .any(|c| c.role == SYSTEM_ROLE && c.content.contains("without commit")),
        "Expected a SYSTEM_ROLE comment explaining why no commit was made"
    );
}

/// handle_qa_passed with untracked files present should claim the ticket
/// to InSanitation and dispatch a sanitation agent. Creates a real git repo
/// with an untracked file to exercise the full claim path.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn handle_qa_passed_untracked_files_to_insanitation() {
    // Skip if git is not installed — the test cannot create a repo.
    if !crate::git_commands::git_is_installed().await {
        eprintln!("git not installed — skipping git-dependent test");
        return;
    }

    // Create a temp directory and init a git repo
    let (_dir, repo_path) = crate::util::test::init_temp_repo();

    // Create an untracked file
    std::fs::write(repo_path.join("untracked.txt"), b"garbage").expect("write untracked file");

    let (ws, ticket_id) = setup_ticket(
        repo_path.to_str().unwrap(),
        "qa_untracked",
        "QA Untracked",
        TicketPhase::QaPassed,
    )
    .await;

    let ticket = expect_ticket(board(), &ticket_id).await;

    handle_qa_passed(ticket, ws).await;

    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::InSanitation,
        "QA passed with untracked files should transition to InSanitation"
    );

    // Verify assigned_to is set to a sanitation agent ID for this ticket.
    //
    // `handle_qa_passed` claims with the unsuffixed base ID
    // (`ticket_{id}_sanitation`); the spawned dispatch task may then overwrite
    // it with the suffixed registered ID (`ticket_{id}_sanitation_{suffix}`)
    // before this assertion runs. Both are valid sanitation assignments, so
    // match the prefix rather than the exact claim-time value — asserting the
    // exact unsuffixed ID would be racy with the background dispatch.
    let ticket = expect_ticket(board(), &ticket_id).await;
    let base_key = crate::session::ticket_agent_id(&ticket_id, crate::Role::Sanitation.as_str());
    assert!(
        ticket
            .assigned_to
            .as_deref()
            .is_some_and(|a| a.starts_with(&base_key)),
        "assigned_to should be a sanitation agent ID for this ticket, got {:?}",
        ticket.assigned_to,
    );
}

// ── Sanitation agent registration / comment routing wiring ──

/// Sanitation dispatch must persist the SAME suffixed agent ID it registers
/// with the message router — the routing contract.
///
/// The board routes mid-work comments to the exact ID stored in
/// `assigned_to` (`route_comment_to_agents` → `try_route`).
/// [`BoardStore::claim_sanitation`] stores the unsuffixed base ID as a
/// placeholder; [`register_agent_and_assign`] (called by `run_stage_agent_round`)
/// must overwrite it with the suffixed ID the run actually registers,
/// otherwise comments are silently dropped for the whole phase.
///
/// This test exercises the register+assign scaffolding directly (no LLM
/// involved — `dispatch_sanitation` itself would invoke a real provider) and
/// verifies end-to-end delivery through `add_comment` → message router.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn sanitation_register_persists_registered_id() {
    init_management_test_stores().await;
    let ws = test_ws_named("/tmp/test_san_register", "ws_san_register");
    let ticket_id = make_ticket(board(), &ws, "San Register", TicketPhase::InSanitation).await;

    // Mirror the run_stage_agent_round scaffolding: job-derived agent ID first,
    // then register + persist the same ID in assigned_to.
    let agent_id = "ticket_test-job_sanitation".to_string();
    let mut rx = register_agent_and_assign(
        &ticket_id,
        &agent_id,
        "Failed to persist assigned_to for sanitation agent — mid-run comments may not route",
    )
    .await;

    // The stored ID must be exactly the ID registered in the router — the
    // mismatch that broke comment routing.
    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.assigned_to.as_deref(),
        Some(agent_id.as_str()),
        "assigned_to must store the exact suffixed agent ID registered in the router"
    );

    // The ID must be run-unique via the job id (fresh session per run —
    // agent id `ticket_{job_id}_sanitation`).
    assert_eq!(
        agent_id,
        "ticket_test-job_sanitation".to_string(),
        "sanitation agent ID must be job-derived for run isolation, got {agent_id}"
    );

    // Wiring proof: a comment routed to the assigned sanitation agent is
    // delivered via the message router — no silent drop.
    board()
        .add_comment(&ticket_id, "manager", "mid-run ping")
        .await
        .expect("add_comment should succeed");
    let job = rx
        .try_recv()
        .expect("comment routed to the assigned sanitation agent should be delivered");
    assert_eq!(job.content, "mid-run ping");
    assert_eq!(job.kind, crate::message_router::JobKind::TicketComment);

    message_router::unregister_agent(&agent_id);
}

/// Pin the mismatch shape: an unsuffixed `assigned_to` (the
/// `claim_sanitation` placeholder) does NOT match the suffixed agent ID a
/// sanitation run registers — comments are silently dropped.
///
/// This is the regression the fix eliminates: [`register_agent_and_assign`]
/// overwrites the placeholder with the registered suffixed ID so routing
/// matches. The negative assertion guards against someone "simplifying" the
/// fix by dropping the suffix instead (which would regress per-run session
/// isolation).
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn sanitation_unsuffixed_assignment_is_not_routed() {
    init_management_test_stores().await;
    let ws = test_ws_named("/tmp/test_san_mismatch", "ws_san_mismatch");
    let ticket_id = make_ticket(board(), &ws, "San Mismatch", TicketPhase::InSanitation).await;

    // Simulate the pre-fix state: assigned_to holds the unsuffixed base ID
    // (what claim_sanitation stores) while the run registers the suffixed ID.
    let base = crate::session::ticket_agent_id(&ticket_id, crate::Role::Sanitation.as_str());
    board()
        .set_assigned_to_no_cancel(&ticket_id, Some(&base))
        .await
        .expect("set assigned_to");
    let suffixed = format!("{base}_{}", crate::generate_suffix());
    let mut rx = message_router::register_agent(&suffixed);

    board()
        .add_comment(&ticket_id, "manager", "should be dropped")
        .await
        .expect("add_comment should succeed");

    // The comment is routed to the unsuffixed ID (not registered) — dropped.
    assert!(
        rx.try_recv().is_err(),
        "comment routed to an unsuffixed assigned_to must NOT reach the suffixed \
         registered agent (mismatch shape pinned by mahbot-1035)"
    );

    message_router::unregister_agent(&suffixed);
}

/// handle_qa_passed with a clean working tree (no untracked files, no
/// modifications) should transition to Done directly without creating a
/// commit — exercising the clean-tree path through [`finalize_ticket_with_git_status`].
///
/// Creates a real git repo with a clean working tree to exercise the
/// QaPassed→Done transition through the clean-tree path.
#[tokio::test]
async fn handle_qa_passed_clean_tree_to_done() {
    // Skip if git is not installed — the test cannot create a repo.
    if !crate::git_commands::git_is_installed().await {
        eprintln!("git not installed — skipping git-dependent test");
        return;
    }

    let (_dir, repo_path) = crate::util::test::init_temp_repo();

    let (ws, ticket_id) = setup_ticket(
        repo_path.to_str().unwrap(),
        "qa_clean",
        "QA Clean Tree",
        TicketPhase::QaPassed,
    )
    .await;

    let ticket = expect_ticket(board(), &ticket_id).await;

    handle_qa_passed(ticket, ws).await;

    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::Done,
        "QA passed with clean tree should transition to Done"
    );

    // Verify a SYSTEM_ROLE comment was written explaining the clean-tree skip.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert!(
        comments
            .iter()
            .any(|c| c.role == SYSTEM_ROLE && c.content.contains("Clean working tree")),
        "Expected a SYSTEM_ROLE comment explaining the clean-tree skip"
    );
}

// ── process_sanitation_verdict — verdict processing ──────────────────

/// Verify [`process_sanitation_verdict`] across all scenarios:
/// - pass=true, clean → SanitationPassed, no marker comment
/// - pass=false, garbage → ReadyForDevelopment with pipeline reservation and marker comment
/// - pass=true, reviewed files → SanitationPassed with "(files reviewed)" suffix, no marker comment
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn process_sanitation_verdict_cases() {
    /// All scenarios of [`process_sanitation_verdict`]. The two comment-marker
    /// fields use different types: [`Case::sanit_markers`] is `&[&str]`
    /// because a Sanitation role comment is *always* created; [`Case::sys_markers`]
    /// is `Option<Vec<&'static str>>` because a [`SANITATION_ROLE`] failure-marker
    /// comment is *conditional* (only appears on `pass=false`) and its marker value
    /// is loaded from a prompt file at runtime.
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        verdict: crate::SanitationVerdict,
        expected_phase: TicketPhase,
        expected_pipeline_reservation: bool,
        /// Substrings required in a Sanitation role comment (empty = just exists).
        sanit_markers: &'static [&'static str],
        /// Substrings required in a [`SANITATION_ROLE`] comment (the marker
        /// comment written when sanitation fails). `None` = no marker comment.
        sys_markers: Option<&'static [&'static str]>,
    }

    init_management_test_stores().await;

    let clean = crate::SanitationVerdict {
        pass: true,
        garbage_files: vec![],
        rationale: "All files are legitimate project files.".into(),
    };
    let garbage = crate::SanitationVerdict {
        pass: false,
        garbage_files: vec!["node_modules/".into(), "tmp/scratch.js".into()],
        rationale: "These are intermediate build artifacts.".into(),
    };
    let reviewed = crate::SanitationVerdict {
        pass: true,
        garbage_files: vec!["generated/bundle.js".into()],
        rationale: "Reviewed, no issues found.".into(),
    };

    let cases = [
        Case {
            name: "pass=true → SanitationPassed",
            ws_suffix: "sp",
            verdict: clean,
            expected_phase: TicketPhase::SanitationPassed,
            expected_pipeline_reservation: false,
            sanit_markers: &[],
            sys_markers: None,
        },
        Case {
            name: "pass=false → ReadyForDevelopment",
            ws_suffix: "sf",
            verdict: garbage,
            expected_phase: TicketPhase::ReadyForDevelopment,
            expected_pipeline_reservation: true,
            sanit_markers: &["node_modules/"],
            sys_markers: None,
        },
        Case {
            name: "pass=true with reviewed files → SanitationPassed (files reviewed)",
            ws_suffix: "sp_r",
            verdict: reviewed,
            expected_phase: TicketPhase::SanitationPassed,
            expected_pipeline_reservation: false,
            sanit_markers: &["(files reviewed)"],
            sys_markers: None,
        },
    ];

    for case in &cases {
        let ws = test_ws_named("/tmp/test", case.ws_suffix);
        let id = make_ticket(board(), &ws, case.name, TicketPhase::InSanitation).await;
        let ticket = expect_ticket(board(), &id).await;
        process_sanitation_verdict(&ticket, case.verdict.clone()).await;

        let phase = expect_ticket_phase(board(), &id).await;
        assert_eq!(
            phase, case.expected_phase,
            "case {}: expected phase {:?}, got {:?}",
            case.name, case.expected_phase, phase,
        );

        let ticket = expect_ticket(board(), &id).await;
        assert_eq!(
            ticket.pipeline_reservation, case.expected_pipeline_reservation,
            "case {}: pipeline_reservation mismatch",
            case.name,
        );
        assert!(
            ticket.assigned_to.is_none(),
            "case {}: assigned_to should be cleared",
            case.name,
        );

        let comments = board().get_comments(&id).await.expect("get_comments");

        // Sanitation role check
        assert!(
            comments.iter().any(|c| c.role == Role::Sanitation.as_str()
                && case.sanit_markers.iter().all(|&m| c.content.contains(m))),
            "case {}: expected Sanitation comment matching {:?}",
            case.name,
            case.sanit_markers,
        );

        // Marker role comment check (written with SANITATION_ROLE on pass=false)
        match &case.sys_markers {
            Some(markers) => {
                assert!(
                    comments.iter().any(|c| c.role == SANITATION_ROLE
                        && markers.iter().all(|&m| c.content.contains(m))),
                    "case {}: expected SANITATION_ROLE comment matching {:?}",
                    case.name,
                    markers,
                );
            }
            None => assert!(
                !comments.iter().any(|c| c.role == SANITATION_ROLE),
                "case {}: expected no SANITATION_ROLE comment",
                case.name,
            ),
        }
    }
}

/// Verify [`dispatch_diagnostics`] behaviour across all scenarios:
///
/// | Scenario | Commands | Expected Phase | Pipeline Reservation | Comment Contains |
/// |---|---|---|---|---|
/// | No diagnostics commands | None (unset) | DiagnosticsDone | false | "No diagnostics commands are configured" |
/// | Diagnostics failure | `false` | ReadyForDevelopment | true | `diagnostics_failed.md` marker |
/// | Diagnostics pass | `true`, ... | DiagnosticsDone | false | `diagnostics_passed.md` marker |
/// | DB error (corrupt JSON) | N/A (corrupt) | DiagnosticsDone | false | "database error" |
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn dispatch_diagnostics_cases() {
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        title: &'static str,
        /// Diagnostics commands to persist (None = leave unset).
        commands: Option<DiagnosticsCommands>,
        /// If true, overwrite the diagnostics column with invalid JSON.
        corrupt_diagnostics: bool,
        /// If true, create a real temp directory for command execution.
        needs_tempdir: bool,
        expected_phase: TicketPhase,
        expected_pipeline_reservation: bool,
        /// Substrings that must all be present in a DIAGNOSTICS_ROLE comment.
        expected_comment_contains: &'static [&'static str],
    }

    const NO_DIAG_CMDS: &[&str] = &["No diagnostics commands are configured"];
    const DB_ERR: &[&str] = &["database error"];

    init_management_test_stores().await;

    let diagnostics_failed_marker: &'static str =
        load_prompt("pipeline/diagnostics_failed.md").leak();
    let diagnostics_passed_marker: &'static str =
        load_prompt("pipeline/diagnostics_passed.md").leak();

    let fail_comment_contains: &'static [&'static str] =
        Box::leak(vec![diagnostics_failed_marker].into_boxed_slice());
    let pass_comment_contains: &'static [&'static str] =
        Box::leak(vec![diagnostics_passed_marker, "PASSED in"].into_boxed_slice());

    let fail_cmds = DiagnosticsCommands {
        format: Some("false".to_string()),
        ..Default::default()
    };
    let pass_cmds = DiagnosticsCommands {
        format: Some("true".to_string()),
        type_check: Some("true".to_string()),
        ..Default::default()
    };

    let cases = [
        Case {
            name: "no diagnostics commands",
            ws_suffix: "dc_no_cmds",
            title: "No Diagnostics Commands",
            commands: None,
            corrupt_diagnostics: false,
            needs_tempdir: false,
            expected_phase: TicketPhase::DiagnosticsDone,
            expected_pipeline_reservation: false,
            expected_comment_contains: NO_DIAG_CMDS,
        },
        Case {
            name: "diagnostics failure",
            ws_suffix: "dc_fail",
            title: "Diagnostics Failure Test",
            commands: Some(fail_cmds),
            corrupt_diagnostics: false,
            needs_tempdir: true,
            expected_phase: TicketPhase::ReadyForDevelopment,
            expected_pipeline_reservation: true,
            expected_comment_contains: fail_comment_contains,
        },
        Case {
            name: "diagnostics all pass",
            ws_suffix: "dc_pass",
            title: "Diagnostics All Pass Test",
            commands: Some(pass_cmds),
            corrupt_diagnostics: false,
            needs_tempdir: true,
            expected_phase: TicketPhase::DiagnosticsDone,
            expected_pipeline_reservation: false,
            expected_comment_contains: pass_comment_contains,
        },
        Case {
            name: "diagnostics DB error",
            ws_suffix: "dc_db_err",
            title: "Diagnostics DB Error Test",
            commands: None,
            corrupt_diagnostics: true,
            needs_tempdir: false,
            expected_phase: TicketPhase::DiagnosticsDone,
            expected_pipeline_reservation: false,
            expected_comment_contains: DB_ERR,
        },
    ];

    for case in &cases {
        let (_dir, ws_path): (Option<tempfile::TempDir>, String) = if case.needs_tempdir {
            let dir = tempfile::tempdir().expect("create temp dir");
            let path = dir.path().to_string_lossy().to_string();
            (Some(dir), path)
        } else {
            (None, format!("/tmp/{}", case.ws_suffix))
        };

        let ws = create_test_workspace(&ws_path, case.ws_suffix).await;

        if let Some(cmds) = &case.commands {
            crate::workspace::store()
                .set_diagnostics(case.ws_suffix, cmds)
                .await
                .expect("set diagnostics");
        }
        if case.corrupt_diagnostics {
            crate::workspace::store()
                .conn
                .execute(
                    "UPDATE workspaces SET diagnostics = ?1 WHERE name = ?2",
                    turso::params!["not valid json", case.ws_suffix],
                )
                .await
                .expect("set diagnostics to invalid JSON");
        }

        let ticket_id = make_ticket(board(), &ws, case.title, TicketPhase::InDiagnostics).await;

        // NOTE: Do NOT claim the ticket beforehand — dispatch_diagnostics
        // calls claim_diagnostics internally as its first step.
        let ticket = expect_ticket(board(), &ticket_id).await;
        dispatch_diagnostics(Arc::new(ticket), ws).await;

        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase, case.expected_phase,
            "case {}: expected phase {:?}, got {:?}",
            case.name, case.expected_phase, phase,
        );

        let ticket = expect_ticket(board(), &ticket_id).await;
        assert_eq!(
            ticket.pipeline_reservation, case.expected_pipeline_reservation,
            "case {}: pipeline_reservation mismatch",
            case.name,
        );
        assert!(
            ticket.assigned_to.is_none(),
            "case {}: assigned_to should be cleared after diagnostics dispatch",
            case.name,
        );

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert!(
            !comments.is_empty(),
            "case {}: should have written at least one comment",
            case.name,
        );
        let has_expected = comments.iter().any(|c| {
            c.role == DIAGNOSTICS_ROLE
                && case
                    .expected_comment_contains
                    .iter()
                    .all(|&marker| c.content.contains(marker))
        });
        assert!(
            has_expected,
            "case {}: should have a DIAGNOSTICS_ROLE comment containing: {:?}",
            case.name, case.expected_comment_contains,
        );
    }
}

// ── dispatch_verifiers skip-review ──────────────────────────────

/// When the current content is identical to the ticket's recorded reviewed
/// base (same HEAD, same index tree, clean porcelain), the reviewer pass may
/// legitimately be skipped — this is the comment-only-round case.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn dispatch_verifiers_skip_review_when_content_matches_base() {
    if !crate::git_commands::git_is_installed().await {
        eprintln!("git not installed — skipping git-dependent test");
        return;
    }

    let (_dir, repo_path) = crate::util::test::init_temp_repo();
    let repo_str = repo_path.to_str().expect("temp path is valid UTF-8");

    let (ws, ticket_id) = setup_ticket(
        repo_str,
        "skip_review_content_test",
        "Skip Review Content Test",
        TicketPhase::InReview,
    )
    .await;

    // Record a reviewed base matching the current repo content
    // (HEAD + index tree of the clean committed tree).
    let head = crate::git_commands::run_git_head(&repo_path)
        .await
        .expect("repo has commits");
    let tree = crate::git_commands::run_git_write_tree(&repo_path)
        .await
        .expect("index writable");
    board()
        .set_reviewed_base(&ticket_id, Some(&head), Some(&tree))
        .await
        .expect("set_reviewed_base");

    let ticket = Arc::new(expect_ticket(board(), &ticket_id).await);

    dispatch_verifiers(ticket, ws, REVIEWER_VI).await;

    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::Reviewed,
        "Content identical to the reviewed base should skip review and go directly to Reviewed"
    );

    // Verify a SYSTEM_ROLE comment was written explaining the skip.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert!(
        comments
            .iter()
            .any(|c| c.role == SYSTEM_ROLE && c.content.contains("Skipping reviewer dispatch")),
        "Expected a SYSTEM_ROLE comment explaining the skip-review reason"
    );
}

/// The skip decision must be conservative — may over-review, never
/// under-review: only content identical to the recorded base skips.
#[test]
fn should_skip_review_decision_matrix() {
    let base = (Some("base-head"), Some("base-tree"));
    let same = base;
    let none = (None, None);
    #[expect(clippy::type_complexity)] // skip-review decision matrix
    let cases: [(
        (Option<&str>, Option<&str>),
        (Option<&str>, Option<&str>),
        &str,
        bool,
    ); 12] = [
        // No recorded base → never skip (first review round).
        (none, same, "", false),
        ((None, Some("base-tree")), same, "", false),
        ((Some("base-head"), None), same, "", false),
        // Identical content → skip (comment-only round).
        (base, same, "", true),
        // New committed content (HEAD change) → review.
        (base, (Some("new-head"), Some("new-tree")), "", false),
        // Empty commit (HEAD change, same tree) → review.
        (base, (Some("new-head"), Some("base-tree")), "", false),
        // Staged content (index tree change) → review.
        (base, (Some("base-head"), Some("new-tree")), "", false),
        // Unstaged / untracked changes → review.
        (base, same, " M src/lib.rs", false),
        (base, same, "?? new_file.txt", false),
        // Uncomputable identity → review (fail open).
        (base, (None, Some("base-tree")), "", false),
        (base, (Some("base-head"), None), "", false),
        (base, same, "\n\n", true), // blank porcelain is still clean
    ];
    for (i, (reviewed, current, porcelain, expected)) in cases.iter().enumerate() {
        assert_eq!(
            should_skip_review(reviewed.0, reviewed.1, current.0, current.1, porcelain),
            *expected,
            "case {i}"
        );
    }
}

// ── Joint-comment raw dumps ────────────────────────────────────────────

/// Build a [`crate::retry::RetryExhausted`] with the given last-attempt raw text.
fn retry_exhausted_with_raw(last_raw: Option<String>) -> crate::retry::RetryExhausted {
    let rec = crate::retry::RetryFailureRecord::new_simple(
        crate::retry::FailureClass::Parse,
        &anyhow::anyhow!("parse failed"),
        None,
    );
    crate::retry::RetryExhausted::with_last_raw(
        vec![rec],
        crate::retry::FailureClass::Parse,
        last_raw,
    )
}

/// The joint comment's "Agent failures" appendix carries the raw last-attempt
/// response for parse-failed agents and the collapsed reason for no-response
/// agents — replacing the old per-agent failure comments.
#[test]
fn joint_comment_includes_failed_agent_dumps() {
    let raw = r#"{"score": 9, "critique": "solid", "issues": []}"#;
    let round = crate::joint_verdict::JointRound {
        stage: "Review",
        dispatched: 2,
        verdicts: vec![],
        failures: vec![
            crate::joint_verdict::JointFailure {
                agent_index: 0,
                dump: crate::util::scrub_credentials(&raw_response_dump_section(
                    &retry_exhausted_with_raw(Some(raw.to_string())),
                )),
            },
            crate::joint_verdict::JointFailure {
                agent_index: 1,
                dump: "agent produced no response".to_string(),
            },
        ],
        header: String::new(),
        threshold: 9,
    };
    let comment = crate::joint_verdict::render_joint_comment(
        &round,
        &crate::consensus::RepairOutcome::Fallback,
        &crate::consensus::ItemTable::new(&crate::joint_verdict::issues_by_agent(&round)),
    );
    assert!(
        !comment.contains("valid verdicts"),
        "verifier comments carry no verdict/threshold headline: {comment}"
    );
    assert!(
        comment.contains("Raw agent response"),
        "parse-failed dump marker must appear: {comment}"
    );
    assert!(
        comment.contains(raw),
        "raw text must be in the comment: {comment}"
    );
    assert!(
        comment.contains("agent produced no response"),
        "no-response reason must appear: {comment}"
    );
    assert!(comment.contains("### Agent failures"), "{comment}");
    assert!(
        comment.contains("Agent 2"),
        "agent indices are 1-based: {comment}"
    );
}

#[test]
fn raw_response_dump_section_covers_sanitation_shape() {
    // The sanitation failure comment path reuses raw_response_dump_section;
    // verify the same truncation/marker contract holds there.
    let raw = "Sanitation agent output with details";
    let failure = retry_exhausted_with_raw(Some(raw.to_string()));
    let section = raw_response_dump_section(&failure);
    assert!(section.contains("Raw agent response"), "{section}");
    assert!(section.contains(raw), "{section}");
}

#[test]
fn engineer_failure_comment_classifies_causes() {
    // LLM retry exhaustion — provider/attempt/status detail preserved.
    // Matched via the agent-loop marker ("exhausted retry budget").
    let err = "LLM step failed at iteration 3: LLM call exhausted retry budget: \
               12 attempt(s) failed (last: transport): OpenRouter API error (503): \
               Service is too busy";
    let c = engineer_failure_comment(false, false, Some(err));
    assert!(c.contains("LLM provider retry exhaustion"), "{c}");
    assert!(c.contains("503"), "{c}");

    // Service shutdown — global token checked before the per-agent token.
    let c = engineer_failure_comment(true, true, None);
    assert!(c.contains("service shutting down"), "{c}");

    // User cancellation (/stop) — distinct from shutdown.
    let c = engineer_failure_comment(false, true, None);
    assert!(c.contains("cancelled by user"), "{c}");

    // Concrete agent error — underlying detail persisted.
    let c = engineer_failure_comment(
        false,
        false,
        Some("Agent exceeded maximum of 1000 tool rounds"),
    );
    assert!(
        c.contains("Agent exceeded maximum of 1000 tool rounds"),
        "{c}"
    );

    // Genuinely unknown cause — generic template retained.
    let c = engineer_failure_comment(false, false, None);
    assert!(c.contains("technical issues"), "{c}");
}

#[test]
fn engineer_failure_comment_scrubs_and_truncates() {
    // Provider error bodies can echo request data — scrub before persisting.
    // Assert on the secret tail ("abcdef") rather than the full secret so the
    // assertions stay meaningful even if tooling redacts secret-looking
    // literals from the fixture (mirrors the agent.rs scrubbing test).
    let err = format!(
        "LLM call exhausted retry budget: 12 attempt(s) failed (last: transport): \
         provider=x attempt 1/1: retryable; error=api_key={secret} boom",
        secret = "sk-1234567890abcdef"
    );
    let c = engineer_failure_comment(false, false, Some(&err));
    assert!(c.contains("[REDACTED]"), "{c}");
    assert!(
        !c.contains("abcdef"),
        "scrubbing must remove the secret tail: {c}"
    );

    // Oversized detail is sandwich-truncated to the verdict dump cap.
    let big = format!("LLM call exhausted retry budget: {}", "x".repeat(30_000));
    let c = engineer_failure_comment(false, false, Some(&big));
    assert!(
        c.contains("bytes omitted at engineer failure truncation"),
        "{c}"
    );
    assert!(c.len() < 26_000, "comment capped, got {}", c.len());
}

/// The engineer's comment extraction is fail-open and scrubbed on every path:
/// extraction failure and empty/blank item lists fall back to the raw response
/// (credential-scrubbed), while a valid item list renders the compact bullet
/// summary (also scrubbed).
#[tokio::test]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn engineer_comment_text_fail_open_and_renders() {
    use crate::util::test::{
        FakeProvider, install_fake_provider, install_test_retry_policy, retry_tests_lock,
    };

    let _lock = retry_tests_lock();
    let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
    let ws = test_ws_named("/tmp/test_ws", "eng_comment");
    let raw = "raw response with secret=abcdefgh1234";
    let new_agent = |suffix: &str| {
        Agent::new(
            format!("eng_comment_{suffix}_{}", crate::generate_suffix()),
            Role::Engineer,
            &ws,
            None,
            String::new(),
            String::new(),
            None,
            None,
        )
    };

    // Extraction failure → raw response kept (fail-open) and scrubbed.
    let fake = FakeProvider::new()
        .err(crate::retry::FailureClass::Transport, "boom")
        .err(crate::retry::FailureClass::Transport, "boom")
        .err(crate::retry::FailureClass::Transport, "boom");
    let _provider = install_fake_provider(Arc::new(fake));
    let comment = engineer_comment_text(&new_agent("err"), raw).await;
    assert_eq!(
        comment,
        crate::util::scrub_credentials(raw),
        "extraction failure keeps the raw response"
    );
    assert!(
        comment.contains("[REDACTED]") && !comment.contains("abcdefgh1234"),
        "fallback is scrubbed: {comment}"
    );

    // Empty item list → raw response kept.
    let fake = FakeProvider::new().ok(r#"{"items": []}"#);
    let _provider = install_fake_provider(Arc::new(fake));
    let comment = engineer_comment_text(&new_agent("empty"), raw).await;
    assert_eq!(
        comment,
        crate::util::scrub_credentials(raw),
        "empty items fall back to raw"
    );

    // Blank-string items → raw response kept (degenerate model emission).
    let fake = FakeProvider::new().ok(r#"{"items": [""]}"#);
    let _provider = install_fake_provider(Arc::new(fake));
    let comment = engineer_comment_text(&new_agent("blank"), raw).await;
    assert_eq!(
        comment,
        crate::util::scrub_credentials(raw),
        "blank items fall back to raw"
    );

    // Valid items → compact bullet list.
    let fake = FakeProvider::new().ok(r#"{"items": ["implemented X", "fixed Y"]}"#);
    let _provider = install_fake_provider(Arc::new(fake));
    let comment = engineer_comment_text(&new_agent("ok"), raw).await;
    assert_eq!(
        comment, "Implemented / fixed / executed:\n- implemented X\n- fixed Y",
        "valid items render the compact bullet list"
    );
}

// ── Engineer hard-failure bounce ────────────────────────────────────────

/// Build a test Engineer agent for the finalizer tests (registered with the
/// ticket so cancellation-by-ticket works).
fn engineer_finalize_test_agent(ws: &Workspace, ticket: &Ticket, suffix: &str) -> Agent {
    Agent::new(
        format!("eng_finalize_{suffix}_{}", crate::generate_suffix()),
        Role::Engineer,
        ws,
        Some(ticket.clone()),
        String::new(),
        String::new(),
        None,
        None,
    )
}

/// A genuine hard "Agent failed" outcome must NOT fail the ticket: it bounces
/// back to ReadyForDevelopment (sharing the review/QA bounce budget, with the
/// pipeline reservation for rework priority), pauses the workspace, and
/// records the concrete error as a SYSTEM_ROLE comment — so the retry's
/// feedback window (comments after the last engineer-role comment) and the
/// Manager notification both carry it.
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn engineer_hard_failure_bounces_to_ready_for_development() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = create_test_workspace("/tmp/eng_bounce_ws", "ws_eng_bounce").await;
    let ticket_id = make_ticket(board(), &ws, "Eng Hard Failure", TicketPhase::InDevelopment).await;
    let ticket = expect_ticket(board(), &ticket_id).await;
    let mut agent = engineer_finalize_test_agent(&ws, &ticket, "hard");
    agent.failure = Some("OpenRouter 500: service is too busy".to_string());

    finalize_engineer_round(&ticket, &agent, None, "job_eng_bounce", false).await;

    let t = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::ReadyForDevelopment,
        "a hard engineer failure must bounce the ticket, not fail it"
    );
    assert_eq!(
        t.bounce_count, 1,
        "the hard-failure bounce must consume the shared review/QA bounce budget"
    );
    assert!(
        t.pipeline_reservation,
        "the bounce must set rework priority over fresh ReadyForDevelopment tickets"
    );

    let ws_after = crate::workspace::store()
        .get_by_name("ws_eng_bounce")
        .await
        .expect("query workspace")
        .expect("workspace exists");
    assert!(
        ws_after.paused,
        "a hard engineer failure must pause the workspace"
    );

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    let last = comments.last().expect("failure comment written");
    assert_eq!(
        last.role, SYSTEM_ROLE,
        "the failure comment must use SYSTEM_ROLE so the retry feedback window \
         (comments after the last engineer-role comment) includes it"
    );
    assert!(
        last.content.contains("OpenRouter 500: service is too busy"),
        "the concrete error must be recorded on the ticket: {}",
        last.content
    );
    assert!(
        last.content.contains("Workspace paused"),
        "the pause notice must be attached to the failure comment: {}",
        last.content
    );
}

/// The engineer hard-failure bounce trip: when the shared bounce budget is
/// exhausted, the ticket moves to Failed (not RFD) — the workspace is STILL
/// paused, ReadyForDevelopment siblings are NOT drained to Planning (unlike
/// the verifier trip), the counter stays at the max, and both the trip
/// comment and the concrete-error comment are written (trip first, so
/// notify_ticket's last-comment lookup surfaces the error).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn engineer_hard_failure_budget_exhaustion_fails_ticket() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = create_test_workspace("/tmp/eng_trip_ws", "ws_eng_trip").await;
    let ticket_id = make_ticket(board(), &ws, "Eng Trip", TicketPhase::InDevelopment).await;
    // A sibling RFD ticket that must NOT be drained to Planning by the trip.
    let sibling_id = make_ticket(
        board(),
        &ws,
        "Eng Trip Sibling",
        TicketPhase::ReadyForDevelopment,
    )
    .await;
    board()
        .conn
        .execute(
            "UPDATE tickets SET bounce_count = ?1 WHERE id = ?2",
            turso::params![
                i64::try_from(crate::joint_verdict::MAX_BOUNCES).unwrap(),
                ticket_id.as_str()
            ],
        )
        .await
        .expect("set bounce_count to max");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let mut agent = engineer_finalize_test_agent(&ws, &ticket, "trip");
    agent.failure = Some("provider exploded".to_string());

    finalize_engineer_round(&ticket, &agent, None, "job_eng_trip", false).await;

    let t = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::Failed,
        "an engineer hard failure at the bounce budget max must fail the ticket"
    );
    assert_eq!(
        t.bounce_count,
        i64::try_from(crate::joint_verdict::MAX_BOUNCES).unwrap(),
        "the failing bounce is not counted — the counter stays at the max"
    );

    let ws_after = crate::workspace::store()
        .get_by_name("ws_eng_trip")
        .await
        .expect("query workspace")
        .expect("workspace exists");
    assert!(
        ws_after.paused,
        "the budget-exhausting hard failure must pause the workspace too"
    );

    let sibling = expect_ticket(board(), &sibling_id).await;
    assert_eq!(
        sibling.phase,
        TicketPhase::ReadyForDevelopment,
        "the engineer trip must NOT drain ReadyForDevelopment siblings to Planning"
    );

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    let trip_idx = comments
        .iter()
        .position(|c| c.content.contains("circuit breaker"))
        .expect("trip comment written");
    let failure_idx = comments
        .iter()
        .rposition(|c| c.content.contains("provider exploded"))
        .expect("failure comment written");
    assert!(
        trip_idx < failure_idx,
        "the trip comment must be written BEFORE the failure comment so the \
         notification's last-comment lookup surfaces the concrete error"
    );
    assert_eq!(
        comments[failure_idx].role, SYSTEM_ROLE,
        "the failure comment must be SYSTEM_ROLE"
    );
}

/// A user-initiated cancellation of the engineer keeps today's semantics:
/// ticket Failed + workspace paused, never auto-re-queued (no bounce).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn engineer_cancel_fails_ticket_without_bounce() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = create_test_workspace("/tmp/eng_cancel_ws", "ws_eng_cancel").await;
    let ticket_id = make_ticket(board(), &ws, "Eng Cancel", TicketPhase::InDevelopment).await;
    let ticket = expect_ticket(board(), &ticket_id).await;
    let agent = engineer_finalize_test_agent(&ws, &ticket, "cancel");
    crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket_id);

    finalize_engineer_round(&ticket, &agent, None, "job_eng_cancel", false).await;

    let t = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::Failed,
        "a user-cancelled engineer run must fail the ticket, not bounce it"
    );
    assert_eq!(
        t.bounce_count, 0,
        "a user cancel must not consume the bounce budget"
    );

    let ws_after = crate::workspace::store()
        .get_by_name("ws_eng_cancel")
        .await
        .expect("query workspace")
        .expect("workspace exists");
    assert!(
        ws_after.paused,
        "a user cancel must pause the workspace (unchanged)"
    );

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    let last = comments.last().expect("failure comment written");
    assert!(
        last.content.contains("cancelled by user"),
        "the cancel cause must be recorded: {}",
        last.content
    );
}

/// A drain-cut engineer round (response None + graceful drain active) must
/// leave the job 'launched' for boot resume — no bounce, no Failed
/// transition, no failure comment, no pause (requirement 4). Uses a REAL
/// launched job row so the caller's skip of job terminalization is observed:
/// a bug that completed the job on the drain bail would flip it to 'done'.
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn engineer_failure_during_drain_stays_queued_for_boot_resume() {
    init_management_test_stores().await;
    let _lock = crate::util::test::retry_tests_lock();

    let ws = create_test_workspace("/tmp/eng_drain_ws", "ws_eng_drain").await;
    let ticket_id = make_ticket(board(), &ws, "Eng Drain", TicketPhase::InDevelopment).await;
    let job_id = "job_eng_drain";
    let now = crate::turso::now();
    JobRowBuilder::new(
        &crate::session::store().conn,
        job_id,
        "ticket_stage",
        "engineer",
        &ws.name,
    )
    .timestamps(now)
    .insert()
    .await
    .expect("insert launched job row");
    let ticket = expect_ticket(board(), &ticket_id).await;
    let mut agent = engineer_finalize_test_agent(&ws, &ticket, "drain");
    agent.failure = Some("drained boom".to_string());

    crate::shutdown::drain_begin();
    finalize_engineer_round(&ticket, &agent, None, job_id, false).await;
    // Clear the process-global drain flag BEFORE the assertions so a failing
    // assert cannot poison the serialized group.
    crate::shutdown::drain_clear();

    let t = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::InDevelopment,
        "a drain-cut engineer round must NOT transition the ticket — it stays queued for boot resume"
    );
    assert_eq!(
        t.bounce_count, 0,
        "a drain-cut round must not consume the bounce budget"
    );
    let ws_after = crate::workspace::store()
        .get_by_name("ws_eng_drain")
        .await
        .expect("query workspace")
        .expect("workspace exists");
    assert!(
        !ws_after.paused,
        "a drain-cut round must not pause the workspace"
    );
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    assert!(
        comments.is_empty(),
        "no failure comment during the drain (the job resumes at boot)"
    );
    let status: Option<String> = crate::session::store()
        .conn
        .query_optional(
            "SELECT status FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
            |row| row.get::<String>(0),
        )
        .await
        .expect("query job status");
    assert_eq!(
        status.as_deref(),
        Some("launched"),
        "the drain-cut job must stay 'launched' for boot resume — never terminalized to 'done'"
    );
}

/// The post-pause drain guard inside [`handle_engineer_failure`] — the second
/// aborting check, after the workspace-pause await (the first real gap after
/// the caller's initial [`stage_round_drain_cut`]) — must bail without any
/// side effects and return `false`, so the caller leaves the job
/// status='launched' for boot resume: no bounce, no Failed transition, no
/// comment, no pause. The test drives the drain pre-active so the guard's
/// bail path is exercised directly (the pause itself is skipped — shutdown is
/// excluded inside [`pause_workspace_on_failure`]).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn engineer_failure_post_pause_drain_bails_leaves_job_launched() {
    init_management_test_stores().await;
    let _lock = crate::util::test::retry_tests_lock();

    let ws = create_test_workspace("/tmp/eng_midtail_ws", "ws_eng_midtail").await;
    let ticket_id = make_ticket(
        board(),
        &ws,
        "Eng Midtail Drain",
        TicketPhase::InDevelopment,
    )
    .await;
    let job_id = "job_eng_midtail";
    let now = crate::turso::now();
    JobRowBuilder::new(
        &crate::session::store().conn,
        job_id,
        "ticket_stage",
        "engineer",
        &ws.name,
    )
    .timestamps(now)
    .insert()
    .await
    .expect("insert launched job row");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let mut agent = engineer_finalize_test_agent(&ws, &ticket, "midtail");
    agent.failure = Some("drained boom".to_string());

    crate::shutdown::drain_begin();
    let bailed = handle_engineer_failure(&ticket, &agent, false).await;
    // Clear the process-global drain flag BEFORE the assertions so a failing
    // assert cannot poison the serialized group.
    crate::shutdown::drain_clear();

    assert!(
        !bailed,
        "the mid-tail drain must report the bail so the caller keeps the job launched"
    );
    let t = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::InDevelopment,
        "the mid-tail drain bail must not transition the ticket"
    );
    assert_eq!(
        t.bounce_count, 0,
        "the mid-tail drain bail must not consume the bounce budget"
    );
    let ws_after = crate::workspace::store()
        .get_by_name("ws_eng_midtail")
        .await
        .expect("query workspace")
        .expect("workspace exists");
    assert!(
        !ws_after.paused,
        "the mid-tail drain bail must not pause the workspace"
    );
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    assert!(
        comments.is_empty(),
        "no failure comment on the mid-tail drain bail"
    );
    let status: Option<String> = crate::session::store()
        .conn
        .query_optional(
            "SELECT status FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
            |row| row.get::<String>(0),
        )
        .await
        .expect("query job status");
    assert_eq!(
        status.as_deref(),
        Some("launched"),
        "the mid-tail drain bail must leave the job 'launched' for boot resume"
    );
}

// ── ticket_stage roster helpers ──────────────────────────────────────────
// These pin the agent-id / angle-cycling contract shared by
// `spawn_ticket_stage_round` (fresh dispatch) and `append_ticket_stage_slots`
// (analysis escalation). The fresh-dispatch paths funnel their job+child-row
// spawn tail through `spawn_ticket_stage_job`. The helpers are the single
// home for both rules — if the shape ever changes, these tests are the first
// to notice.

/// The agent-id helper must produce the exact documented shape
/// `ticket_{ticket_id}_{idx}_{suffix}_{role}` for both dispatch paths.
#[test]
fn ticket_stage_agent_id_format() {
    assert_eq!(
        ticket_stage_agent_id("t-42", 0, "abc123", Role::Analyst),
        "ticket_t-42_0_abc123_analyst",
        "base-round slot 0"
    );
    assert_eq!(
        ticket_stage_agent_id("t-42", 2, "abc123", Role::Analyst),
        "ticket_t-42_2_abc123_analyst",
        "base-round slot 2"
    );
    // Escalation continues at the roster length (3, 4) with a FRESH suffix.
    assert_eq!(
        ticket_stage_agent_id("t-42", 3, "def456", Role::Analyst),
        "ticket_t-42_3_def456_analyst",
        "escalation slot 3"
    );
    assert_eq!(
        ticket_stage_agent_id("t-42", 4, "def456", Role::Analyst),
        "ticket_t-42_4_def456_analyst",
        "escalation slot 4"
    );
    // Role string is the canonical lowercase `as_str()` (role LAST).
    assert_eq!(
        ticket_stage_agent_id("t-7", 0, "xyz789", Role::Reviewer),
        "ticket_t-7_0_xyz789_reviewer"
    );
    assert_eq!(
        ticket_stage_agent_id("t-7", 0, "xyz789", Role::Qa),
        "ticket_t-7_0_xyz789_qa"
    );
}

/// The angle-cycling rule must cover all three branches: bare prompt (no
/// angles), join-all for single-slot rounds, and per-index cycling with wrap.
#[test]
fn ticket_stage_slot_task_angle_branches() {
    let prompt = "Review the change";
    let angles = vec!["angle one".to_string(), "angle two".to_string()];

    // No angles → bare prompt untouched (Analyst-style roles).
    assert_eq!(
        ticket_stage_slot_task(prompt, &[], 3, 1),
        prompt,
        "no angles: shared prompt used verbatim"
    );

    // Single-slot round (QA's lone tester) → ALL angle sections joined.
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles, 1, 0),
        format!("{prompt}\n\nangle one\n\nangle two"),
        "slot_count == 1 concatenates every angle section"
    );

    // Multi-slot round → cycling by GLOBAL slot index, wrapping past len.
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles, 2, 0),
        format!("{prompt}\n\nangle one"),
        "global idx 0 → first angle"
    );
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles, 2, 1),
        format!("{prompt}\n\nangle two"),
        "global idx 1 → second angle"
    );
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles, 2, 2),
        format!("{prompt}\n\nangle one"),
        "global idx 2 wraps to first angle"
    );
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles, 2, 3),
        format!("{prompt}\n\nangle two"),
        "global idx 3 wraps to second angle"
    );

    // Escalation view: the job is the whole — a base roster of 3 + 2
    // escalation slots must keep angle selection continuous (idx 3, 4).
    let angles3 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles3, 5, 3),
        format!("{prompt}\n\na"),
        "escalation global idx 3 → angles[3 % 3]"
    );
    assert_eq!(
        ticket_stage_slot_task(prompt, &angles3, 5, 4),
        format!("{prompt}\n\nb"),
        "escalation global idx 4 → angles[4 % 3]"
    );
}

// ── Pending-workspace pickup ─────────────────────────────────────

/// Snapshot the process-global CONFIG and replace it with a controlled state
/// for the duration of the test (pickup gating reads CONFIG).
///
/// Serialized in the `config_persist` group: these tests swap the shared
/// global CONFIG, and an unserialized swap could clobber a concurrent test's
/// CONFIG writes (or be clobbered by them) and fail its asserts
/// nondeterministically.
struct ConfigGuard(crate::config::ConfigData);

impl ConfigGuard {
    fn new(provider_key: Option<&str>, custom_endpoint: Option<&str>) -> Self {
        let snapshot = crate::config::CONFIG.snapshot();
        crate::config::CONFIG.swap(crate::config::ConfigData::STRUCT_FIELDS_DEFAULT);
        if let Some(key) = provider_key {
            let _ = crate::config::CONFIG.set_string_field("provider_key", key);
        }
        if let Some(endpoint) = custom_endpoint {
            let _ = crate::config::CONFIG.set_string_field("provider_endpoint", endpoint);
        }
        Self(snapshot)
    }
}

impl Drop for ConfigGuard {
    fn drop(&mut self) {
        crate::config::CONFIG.swap(self.0.clone());
    }
}

#[tokio::test]
#[serial_test::serial(config_persist)] // swaps the process-global CONFIG
async fn pickup_pending_workspace_waits_for_provider() {
    init_management_test_stores().await;
    // No provider key, default endpoint — the pickup must hold the workspace
    // in pending (no claim, no discovery spawn, no LLM calls).
    let _cfg = ConfigGuard::new(None, None);
    let ws = create_test_workspace("/tmp/test_pickup_wait", "ws_pickup_wait").await;
    assert_eq!(ws.status, WorkspaceStatus::Pending, "precondition: pending");

    pickup_pending_workspace(&ws).await;

    let stored = crate::workspace::store()
        .get_by_name("ws_pickup_wait")
        .await
        .expect("fetch")
        .expect("exists");
    assert_eq!(
        stored.status,
        WorkspaceStatus::Pending,
        "no provider configured → workspace stays pending"
    );
    assert!(
        !stored.paused,
        "no provider → not claimed, the pause toggle is untouched"
    );
}

#[tokio::test]
#[serial_test::serial(config_persist)] // swaps the process-global CONFIG
async fn pickup_claim_claims_when_provider_configured() {
    init_management_test_stores().await;
    let _cfg = ConfigGuard::new(Some("sk-test"), None);
    let ws = create_test_workspace("/tmp/test_pickup_claim", "ws_pickup_claim").await;

    // pickup_claim (the decision + atomic claim half of the pickup) must
    // claim without spawning any discovery task — deterministic, no agents.
    let claimed = pickup_claim(&ws).await;
    let (generation, discover_diagnostics) = claimed.expect("claim should succeed");
    assert_eq!(generation, 0, "fresh workspace has discovery_generation 0");
    assert!(
        discover_diagnostics,
        "no diagnostics exist yet → first discovery must run diagnostics"
    );

    let stored = crate::workspace::store()
        .get_by_name("ws_pickup_claim")
        .await
        .expect("fetch")
        .expect("exists");
    assert_eq!(
        stored.status,
        WorkspaceStatus::Analyzing,
        "provider key configured → pending workspace claimed into discovery"
    );
    assert!(
        stored.paused,
        "the claim must set the analysis pause (blocks pipeline claims while discovery runs)"
    );
}

#[tokio::test]
#[serial_test::serial(config_persist)] // swaps the process-global CONFIG
async fn pickup_claim_claims_without_key_when_custom_endpoint_persisted() {
    init_management_test_stores().await;
    // A keyless custom endpoint counts as provider configured —
    // the runtime honors a persisted custom endpoint, so without an OpenRouter
    // key the pickup must still claim the workspace into discovery.
    let _cfg = ConfigGuard::new(None, Some("http://localhost:8080/v1"));
    let ws = create_test_workspace("/tmp/test_pickup_endpoint", "ws_pickup_endpoint").await;

    let claimed = pickup_claim(&ws).await;
    let (generation, discover_diagnostics) = claimed
        .expect("a persisted custom endpoint without a key must count as provider configured");
    assert_eq!(generation, 0, "fresh workspace has discovery_generation 0");
    assert!(
        discover_diagnostics,
        "no diagnostics exist yet → first discovery must run diagnostics"
    );

    let stored = crate::workspace::store()
        .get_by_name("ws_pickup_endpoint")
        .await
        .expect("fetch")
        .expect("exists");
    assert_eq!(
        stored.status,
        WorkspaceStatus::Analyzing,
        "keyless custom endpoint → pending workspace claimed into discovery"
    );
    assert!(
        stored.paused,
        "the claim must set the analysis pause (blocks pipeline claims while discovery runs)"
    );
}

#[tokio::test]
#[serial_test::serial(config_persist)] // swaps the process-global CONFIG
async fn pickup_pending_workspace_respects_cooldown() {
    init_management_test_stores().await;
    let _cfg = ConfigGuard::new(Some("sk-test"), None);
    let ws = create_test_workspace("/tmp/test_pickup_cooldown", "ws_pickup_cooldown").await;
    crate::workspace::record_pending_pickup_cooldown("ws_pickup_cooldown");

    pickup_pending_workspace(&ws).await;

    let stored = crate::workspace::store()
        .get_by_name("ws_pickup_cooldown")
        .await
        .expect("fetch")
        .expect("exists");
    assert_eq!(
        stored.status,
        WorkspaceStatus::Pending,
        "armed cooldown → pickup must hold the workspace in pending"
    );
    crate::workspace::clear_pending_pickup_cooldown("ws_pickup_cooldown");
}

#[tokio::test]
#[serial_test::serial(config_persist)] // swaps the process-global CONFIG
async fn pickup_skips_non_pending_workspaces() {
    init_management_test_stores().await;
    let _cfg = ConfigGuard::new(Some("sk-test"), None);
    let ws = create_test_workspace("/tmp/test_pickup_skip", "ws_pickup_skip").await;
    crate::workspace::store()
        .set_status("ws_pickup_skip", &WorkspaceStatus::Analyzing)
        .await
        .expect("set status");

    pickup_pending_workspace(&ws).await;

    let stored = crate::workspace::store()
        .get_by_name("ws_pickup_skip")
        .await
        .expect("fetch")
        .expect("exists");
    assert_eq!(
        stored.status,
        WorkspaceStatus::Analyzing,
        "pickup only touches pending workspaces"
    );
}

#[tokio::test]
#[serial_test::serial(config_persist)] // swaps the process-global CONFIG
async fn pickup_claim_is_atomic_against_db_state() {
    init_management_test_stores().await;
    let _cfg = ConfigGuard::new(Some("sk-test"), None);
    let ws = create_test_workspace("/tmp/test_pickup_race", "ws_pickup_race").await;

    // Simulate a concurrent claimer that already transitioned the row: the
    // pickup's in-memory copy still says pending, but the DB says analyzing —
    // the conditional UPDATE must win and no second discovery is spawned.
    crate::workspace::store()
        .claim_pending_for_discovery("ws_pickup_race")
        .await
        .expect("first claim")
        .expect("should claim");

    let claimed = pickup_claim(&ws).await;
    assert!(
        claimed.is_none(),
        "an already-claimed row must not be claimed twice"
    );

    let stored = crate::workspace::store()
        .get_by_name("ws_pickup_race")
        .await
        .expect("fetch")
        .expect("exists");
    assert_eq!(
        stored.status,
        WorkspaceStatus::Analyzing,
        "already-claimed row stays analyzing (single claim)"
    );
}
