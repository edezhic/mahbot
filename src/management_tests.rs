use super::*;
use crate::util::test::make_ticket;
use crate::util::test::{
    create_test_workspace, expect_ticket, expect_ticket_phase, init_management_test_stores,
    init_test_stores,
};
use crate::workspace::test_ws_named;
use strum::IntoEnumIterator;

/// All non-General circuit breaker variants must have a threshold strictly
/// less than [`CircuitBreakerKind::General`]'s threshold.
///
/// ## Rationale
///
/// - **Sanitation breaker** (`threshold = 3`): must trip before the general
///   breaker (`threshold = 30`), otherwise a ticket could accumulate 30+
///   comments during repeated sanitation loops without tripping.
/// - **Diagnostics breaker** (`threshold = 4`): must also trip before the
///   general breaker. This is a conservative approximation — the general
///   breaker counts *all* comments (not just diagnostics), but guaranteeing
///   that diagnostics-only chatter cannot bypass the general breaker prevents
///   pathological ticket growth from repeated diagnostic cycles.

#[test]
fn all_non_general_circuit_breakers_trip_before_general() {
    let general = CircuitBreakerKind::General.threshold();
    for kind in CircuitBreakerKind::iter() {
        if kind == CircuitBreakerKind::General {
            continue;
        }
        assert!(
            kind.threshold() < general,
            "{kind:?}.threshold() ({}) must be less than General.threshold() ({general})",
            kind.threshold(),
        );
    }
}

/// Verify that when the circuit breaker trips on a ticket, all other
/// ReadyForDevelopment tickets in the same workspace are moved to Planning.
/// Tickets in other workspaces must not be affected.
#[tokio::test]
async fn circuit_breaker_moves_other_ready_for_development_tickets_to_planning() {
    init_management_test_stores().await;

    let ws_a = test_ws_named("/ws_a", "ws_a");
    let ws_b = test_ws_named("/ws_b", "ws_b");

    // Create ticket A in workspace A — this will trip the circuit breaker.
    let trip_id = make_ticket(
        board(),
        &ws_a,
        "Trip Ticket",
        TicketPhase::ReadyForDevelopment,
    )
    .await;

    // Create ticket B in workspace A — this should be moved to Planning when A trips.
    let victim_id = make_ticket(
        board(),
        &ws_a,
        "Victim Ticket",
        TicketPhase::ReadyForDevelopment,
    )
    .await;

    // Create ticket C in workspace B — this must NOT be moved.
    let other_ws_id = make_ticket(
        board(),
        &ws_b,
        "Other Workspace Ticket",
        TicketPhase::ReadyForDevelopment,
    )
    .await;

    // Add comments to ticket A so the circuit breaker has something to count
    // (CircuitBreakerKind::General.threshold() + 1 = 31 comments, enough to trip).
    for i in 0..=CircuitBreakerKind::General.threshold() {
        board()
            .add_comment(&trip_id, SYSTEM_ROLE, &format!("Comment {i}"))
            .await
            .expect("add_comment to A");
    }

    // Fetch ticket A and trip the circuit breaker.
    let ticket_a = expect_ticket(board(), &trip_id).await;

    let tripped = try_trip_circuit_breaker(
        &ticket_a,
        TicketPhase::ReadyForDevelopment,
        CircuitBreakerKind::General,
        "test",
    )
    .await;

    assert!(tripped, "circuit breaker should have tripped");

    // ── Verify ticket A is Failed ──
    {
        let ticket_a = expect_ticket(board(), &trip_id).await;
        assert_eq!(
            ticket_a.phase,
            TicketPhase::Failed,
            "tripped ticket A should be Failed"
        );
    }

    // ── Verify ticket B (same workspace) is Planning ──
    {
        let ticket_b = expect_ticket(board(), &victim_id).await;
        assert_eq!(
            ticket_b.phase,
            TicketPhase::Planning,
            "other ReadyForDevelopment ticket B in same workspace should be Planning"
        );
    }

    // ── Verify ticket C (different workspace) is still ReadyForDevelopment ──
    {
        let ticket_c = expect_ticket(board(), &other_ws_id).await;
        assert_eq!(
            ticket_c.phase,
            TicketPhase::ReadyForDevelopment,
            "ticket C in different workspace must not be moved"
        );
    }
}

/// Verify that `record_verdict_comments_tx` correctly writes comments
/// based on verdict filter.
#[tokio::test]
async fn record_verdict_comments_filtering() {
    init_test_stores().await;

    let ticket_id = make_ticket(
        board(),
        &test_ws_named("/tmp/test", "test"),
        "Test",
        TicketPhase::Backlog,
    )
    .await;

    // ── FailingOnly with all-passing verdicts ──
    // Should produce 0 comments (nothing to write).
    let results = vec![pass_result()];
    crate::turso::with_tx(
        &board().conn,
        &ticket_id,
        "test verdict comments",
        async |tx| {
            record_verdict_comments_tx(
                tx,
                &ticket_id,
                &results,
                Role::Reviewer.as_str(),
                VerdictFilter::FailingOnly,
            )
            .await
        },
    )
    .await
    .expect("record_verdict_comments_tx should succeed");

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert_eq!(
        comments.len(),
        0,
        "passing verdicts with FailingOnly filter should produce 0 comments"
    );

    // ── FailingOnly with a failing verdict ──
    // Should produce 1 comment.
    let results = vec![fail_result()];
    crate::turso::with_tx(
        &board().conn,
        &ticket_id,
        "test verdict comments",
        async |tx| {
            record_verdict_comments_tx(
                tx,
                &ticket_id,
                &results,
                Role::Reviewer.as_str(),
                VerdictFilter::FailingOnly,
            )
            .await
        },
    )
    .await
    .expect("record_verdict_comments_tx should succeed");

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert_eq!(
        comments.len(),
        1,
        "failing verdict should create one comment"
    );
    assert_eq!(comments[0].role, "reviewer_1");

    // ── All filter (analyst path) ──
    // Should produce 2 comments (both verdicts recorded).
    let results = vec![
        analyst_verdict(10, "Excellent analysis.", &[]),
        analyst_verdict(4, "Needs more research.", &["Missing citations"]),
    ];
    crate::turso::with_tx(
        &board().conn,
        &ticket_id,
        "test verdict comments",
        async |tx| {
            record_verdict_comments_tx(
                tx,
                &ticket_id,
                &results,
                Role::Analyst.as_str(),
                VerdictFilter::All,
            )
            .await
        },
    )
    .await
    .expect("record_verdict_comments_tx should succeed");

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert_eq!(
        comments.len(),
        3,
        "All filter should write both verdicts (total 3)"
    );
    assert_eq!(comments[1].role, "analyst_1");
    assert_eq!(comments[2].role, "analyst_2");
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
#[tokio::test]
async fn transition_ticket_to_done_buffer_and_notify() {
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
        let t = board()
            .get_ticket(id)
            .await
            .expect("get_ticket")
            .unwrap_or_else(|| panic!("ticket {label} exists"));
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

// ── try_trip_circuit_breaker — sanitation counting ──

/// Verify that the sanitation circuit breaker counting logic works correctly.
#[tokio::test]
async fn sanitation_breaker_counts_failures() {
    init_management_test_stores().await;
    let ticket_id = make_ticket(
        board(),
        &test_ws_named("/tmp/test", "san_breaker_test"),
        "Sanitation Breaker Test",
        TicketPhase::InSanitation,
    )
    .await;

    // Add 2 sanitation failure comments (below threshold of 3).
    for _ in 0..2 {
        add_sanitation_failure(&ticket_id).await;
    }

    let ticket = expect_ticket(board(), &ticket_id).await;

    assert!(
        !try_trip_circuit_breaker(
            &ticket,
            TicketPhase::InSanitation,
            CircuitBreakerKind::Sanitation,
            "Sanitation",
        )
        .await,
        "Should NOT trip with 2 failures (threshold: 3)"
    );

    // Add a 3rd failure comment (should still not trip — reads fresh comments
    // from DB, not the stale in-memory ticket.comments).
    add_sanitation_failure(&ticket_id).await;

    // ... actually 3 <= 3 means the breaker does NOT trip yet.
    // The breaker trips when count > threshold, i.e., at 4 failures.
    // Add a 4th failure.
    add_sanitation_failure(&ticket_id).await;

    // Now with 4 failures, should trip (4 > 3).
    let tripped = try_trip_circuit_breaker(
        &ticket,
        TicketPhase::InSanitation,
        CircuitBreakerKind::Sanitation,
        "Sanitation",
    )
    .await;
    assert!(tripped, "Should trip with 4 failures (threshold: 3, 4 > 3)");

    // Verify the ticket is now Failed
    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::Failed,
        "Circuit breaker should transition to Failed"
    );
}

// ── Setup helpers ──────────────────────────────────────────────────────

/// Shared helper: create a passing verdict (score >= REVIEW_QA_THRESHOLD).
fn pass_verdict() -> crate::Verdict {
    crate::Verdict {
        score: REVIEW_QA_THRESHOLD,
        critique: Some("Good work.".into()),
        issues_detected: vec![],
    }
}

/// Shared helper: create a failing verdict (score < REVIEW_QA_THRESHOLD).
fn fail_verdict() -> crate::Verdict {
    crate::Verdict {
        score: 3,
        critique: Some("Missing error handling.".into()),
        issues_detected: vec!["No timeout check".into()],
    }
}

/// Helper: a `ParallelVerdict` with no response.
fn no_verdict() -> ParallelVerdict {
    ParallelVerdict::NoResponse
}

/// Add a sanitation failure comment for circuit breaker testing.
async fn add_sanitation_failure(ticket_id: &str) {
    let _ = board()
        .add_comment(
            ticket_id,
            SYSTEM_ROLE,
            &format!("{SANITATION_FAILED_PREFIX} — garbage files: 1"),
        )
        .await;
}

/// Helper: wrap a passing verdict (reviewer/QA flow).
fn pass_result() -> ParallelVerdict {
    ParallelVerdict::Verdict(pass_verdict())
}

/// Helper: wrap a failing verdict (reviewer/QA flow).
fn fail_result() -> ParallelVerdict {
    ParallelVerdict::Verdict(fail_verdict())
}

/// Helper: construct an analyst verdict with explicit score / critique / issues.
fn analyst_verdict(score: u8, critique: &str, issues: &[&str]) -> ParallelVerdict {
    ParallelVerdict::Verdict(crate::Verdict {
        score,
        critique: Some(critique.into()),
        issues_detected: issues.iter().map(|&s| s.into()).collect(),
    })
}

// ── process_verifier_verdicts — verdict processing ─────────────────────

/// Verify all verdict-processing outcomes:
/// - All failed → Failed
/// - Any failed → bounce-back to ReadyForDevelopment with pipeline reservation
/// - All passed (Reviewer) → Reviewed
/// - All passed (QA) → QaPassed
#[tokio::test]
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
    }

    init_management_test_stores().await;

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
        },
    ];

    for case in &cases {
        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", case.ws_suffix),
            case.title,
            case.phase,
        )
        .await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        process_verifier_verdicts(&ticket, &case.results, case.vi).await;

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
    }
}

// ── try_trip_circuit_breaker — general circuit breaker ────────

/// Verify the circuit breaker trips at the threshold boundary:
/// - `> CircuitBreakerKind::General.threshold()` comments → trips (ticket → Failed)
/// - `= CircuitBreakerKind::General.threshold()` comments → does NOT trip
///
/// When the breaker trips, also verifies the trip comment contains the
/// "circuit breaker" marker as produced by [`CircuitBreakerKind::comment`].
#[tokio::test]
async fn circuit_breaker_comment_boundary() {
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        title: &'static str,
        comment_count: usize,
        expected_trip: bool,
        expected_phase: TicketPhase,
    }

    init_management_test_stores().await;

    let cases = [
        Case {
            name: "> threshold trips",
            ws_suffix: "cb_thresh",
            title: "CB Threshold",
            comment_count: CircuitBreakerKind::General.threshold() + 1,
            expected_trip: true,
            expected_phase: TicketPhase::Failed,
        },
        Case {
            name: "= threshold does not trip",
            ws_suffix: "cb_no_trip",
            title: "CB No Trip",
            comment_count: CircuitBreakerKind::General.threshold(),
            expected_trip: false,
            expected_phase: TicketPhase::InReview,
        },
    ];

    for case in &cases {
        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", case.ws_suffix),
            case.title,
            TicketPhase::InReview,
        )
        .await;

        for i in 0..case.comment_count {
            board()
                .add_comment(&ticket_id, "user", &format!("Comment {i}"))
                .await
                .expect("add_comment");
        }

        let ticket = expect_ticket(board(), &ticket_id).await;

        let tripped = try_trip_circuit_breaker(
            &ticket,
            TicketPhase::InReview,
            CircuitBreakerKind::General,
            "test",
        )
        .await;
        assert_eq!(
            tripped, case.expected_trip,
            "case {}: expected trip={}, got tripped={}",
            case.name, case.expected_trip, tripped,
        );

        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase, case.expected_phase,
            "case {}: expected phase {:?}, got {:?}",
            case.name, case.expected_phase, phase,
        );

        // When the breaker trips, verify the trip comment contains the
        // circuit breaker marker as produced by `CircuitBreakerKind::comment`.
        if tripped {
            let comments = board()
                .get_comments(&ticket_id)
                .await
                .expect("get_comments");
            let has_marker = comments
                .iter()
                .any(|c| c.content.to_lowercase().contains("circuit breaker"));
            assert!(
                has_marker,
                "case {}: trip comment must contain circuit breaker marker",
                case.name,
            );
        }
    }
}

// ── process_analyst_verdicts — analyst scoring and transitions ─────────

/// Verify process_analyst_verdicts across all outcomes:
/// - All analysts pass → Planning with "All LGTM" summary
/// - Partial fail → Planning with "blockers" summary
/// - No verdicts → Planning with "no analysis" summary
#[tokio::test]
async fn process_analyst_verdicts_cases() {
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        title: &'static str,
        results: Vec<ParallelVerdict>,
        expected_comment_substring: &'static str,
    }

    init_management_test_stores().await;

    let cases = vec![
        Case {
            name: "all pass -> Planning with LGTM",
            ws_suffix: "an_all_pass",
            title: "Analyst All Pass",
            results: vec![
                analyst_verdict(10, "Great analysis.", &[]),
                analyst_verdict(9, "Solid work.", &[]),
                analyst_verdict(8, "Good analysis.", &[]),
            ],
            expected_comment_substring: "All LGTM",
        },
        Case {
            name: "partial fail -> Planning with blockers",
            ws_suffix: "an_partial",
            title: "Analyst Partial Fail",
            results: vec![
                analyst_verdict(10, "Great.", &[]),
                analyst_verdict(3, "Poor analysis.", &["Missing data"]),
                analyst_verdict(8, "Decent.", &["Minor issue"]),
            ],
            expected_comment_substring: "blockers",
        },
        Case {
            name: "no verdicts -> Planning with no analysis",
            ws_suffix: "an_no_v",
            title: "Analyst No Verdicts",
            results: vec![no_verdict(); 3],
            expected_comment_substring: "no analysis",
        },
    ];

    for case in &cases {
        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", case.ws_suffix),
            case.title,
            TicketPhase::Analysis,
        )
        .await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        process_analyst_verdicts(&ticket, &case.results).await;

        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
            TicketPhase::Planning,
            "case {}: expected Planning, got {:?}",
            case.name,
            phase,
        );

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");

        let system = comments
            .iter()
            .find(|c| c.role == SYSTEM_ROLE)
            .unwrap_or_else(|| panic!("case {}: system summary comment should exist", case.name));
        assert!(
            system.content.contains(case.expected_comment_substring),
            "case {}: system comment should contain {:?}, got: {}",
            case.name,
            case.expected_comment_substring,
            system.content,
        );
    }
}

// ── handle_qa_passed — QA → Done path ───────────────────────────────

/// handle_qa_passed first checks whether git is available and whether the
/// workspace path is a git repo. In test environments git may exist, but
/// the workspace path is deliberately not a git repo, so the function
/// transitions directly to Done without committing.
/// This test validates the graceful non-git fallback path.
#[tokio::test]
async fn handle_qa_passed_no_git_to_done() {
    // Use a path that cannot be a git repo regardless of the test
    // environment's current working directory.
    let (ws, ticket_id) = setup_ticket(
        "/nonexistent/mahbot-test-qa-no-git",
        "qa_no_git",
        "QA No Git",
        TicketPhase::QaPassed,
    )
    .await;

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
#[tokio::test]
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

    // Verify assigned_to is set to the sanitation session key
    let ticket = expect_ticket(board(), &ticket_id).await;
    let expected_key =
        crate::session::ticket_session_key(&ticket_id, crate::Role::Sanitation.as_str());
    assert_eq!(
        ticket.assigned_to.as_deref(),
        Some(expected_key.as_str()),
        "assigned_to should be set to sanitation session key"
    );
}

// ── process_sanitation_verdict — verdict processing ──────────────────

/// Verify both branches of `process_sanitation_verdict`:
/// - pass=true → SanitationPassed with a sanitation comment
/// - pass=false → ReadyForDevelopment with pipeline reservation,
///   a sanitation comment, and a system circuit-breaker comment
#[tokio::test]
async fn process_sanitation_verdict_cases() {
    init_management_test_stores().await;

    // Case 1: pass=true → SanitationPassed
    let ws_pass = test_ws_named("/tmp/test", "sv_pass");
    let pass_id = make_ticket(board(), &ws_pass, "SV Pass", TicketPhase::InSanitation).await;
    let ticket_pass = expect_ticket(board(), &pass_id).await;

    let pass_verdict = crate::SanitationVerdict {
        pass: true,
        garbage_files: vec![],
        rationale: "All files are legitimate project files.".into(),
    };

    process_sanitation_verdict(&ticket_pass, pass_verdict).await;

    let phase = expect_ticket_phase(board(), &pass_id).await;
    assert_eq!(
        phase,
        TicketPhase::SanitationPassed,
        "pass=true should transition to SanitationPassed, got {phase:?}",
    );

    // Verify a sanitation comment was added
    let comments = board().get_comments(&pass_id).await.expect("get_comments");
    let has_sanitation_comment = comments.iter().any(|c| c.role == Role::Sanitation.as_str());
    assert!(
        has_sanitation_comment,
        "pass=true should add a sanitation comment",
    );

    // No system comment (only added in fail path)
    let has_system_comment = comments.iter().any(|c| c.role == SYSTEM_ROLE);
    assert!(
        !has_system_comment,
        "pass=true should not add a system comment",
    );

    // Case 2: pass=false → ReadyForDevelopment with pipeline reservation
    let ws_fail = test_ws_named("/tmp/test", "sv_fail");
    let fail_id = make_ticket(board(), &ws_fail, "SV Fail", TicketPhase::InSanitation).await;
    let ticket_fail = expect_ticket(board(), &fail_id).await;

    let fail_verdict = crate::SanitationVerdict {
        pass: false,
        garbage_files: vec!["node_modules/".into(), "tmp/scratch.js".into()],
        rationale: "These are intermediate build artifacts.".into(),
    };

    process_sanitation_verdict(&ticket_fail, fail_verdict).await;

    let ticket = expect_ticket(board(), &fail_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::ReadyForDevelopment,
        "pass=false should bounce back to ReadyForDevelopment, got {:?}",
        ticket.phase,
    );
    assert!(
        ticket.pipeline_reservation,
        "pass=false should set pipeline_reservation=true",
    );

    // Verify a sanitation comment was added about the garbage files
    let comments = board().get_comments(&fail_id).await.expect("get_comments");
    let has_garbage_comment = comments
        .iter()
        .any(|c| c.role == Role::Sanitation.as_str() && c.content.contains("node_modules/"));
    assert!(
        has_garbage_comment,
        "pass=false should have a sanitation comment mentioning garbage files",
    );

    // Verify a system comment with SANITATION_FAILED_PREFIX was added
    let has_system_breaker = comments
        .iter()
        .any(|c| c.role == SYSTEM_ROLE && c.content.contains(SANITATION_FAILED_PREFIX));
    assert!(
        has_system_breaker,
        "pass=false should have a system comment with the circuit breaker prefix",
    );
}

/// Verify that when a workspace has no diagnostics commands configured,
/// `dispatch_diagnostics` writes a no-commands comment and transitions
/// the ticket to `DiagnosticsDone`.
#[tokio::test]
async fn no_diagnostics_commands_skips_to_diagnostics_done() {
    init_management_test_stores().await;

    let ws = create_test_workspace("/tmp/test_no_diag", "test_no_diag").await;

    let ticket_id = make_ticket(
        board(),
        &ws,
        "No Diagnostics Commands",
        TicketPhase::InDiagnostics,
    )
    .await;

    // NOTE: Do NOT claim the ticket beforehand — dispatch_diagnostics
    // calls claim_diagnostics internally as its first step.
    let ticket = expect_ticket(board(), &ticket_id).await;

    dispatch_diagnostics(Arc::new(ticket), ws).await;

    // Verify transition to DiagnosticsDone.
    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::DiagnosticsDone,
        "ticket should be DiagnosticsDone when no diagnostics commands are configured",
    );

    // Verify a diagnostics-role comment was written explaining the skip.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert!(
        !comments.is_empty(),
        "should have written at least one comment",
    );
    let comment = &comments[0];
    assert_eq!(comment.role, DIAGNOSTICS_ROLE);
    assert!(
        comment
            .content
            .contains("No diagnostics commands are configured"),
        "comment should explain that no diagnostics commands are configured: {}",
        comment.content,
    );
}

/// Verify that when diagnostics commands fail, `dispatch_diagnostics` writes a
/// failure comment and bounces the ticket back to `ReadyForDevelopment` with
/// `pipeline_reservation = true`. This exercises Path C2 through the complete
/// transaction (comment + transition), complementing the Path B test above
/// and verifying the with_tx crash-consistency fix.
#[tokio::test]
async fn diagnostics_failure_bounces_to_ready_for_development() {
    init_management_test_stores().await;

    let dir = tempfile::tempdir().expect("create temp dir");
    let ws_path = dir.path().to_string_lossy().to_string();
    let ws_name = "test_diag_fail";
    let ws = create_test_workspace(&ws_path, ws_name).await;

    // Set a diagnostics command that will always fail (exit non-zero).
    let cmds = DiagnosticsCommands {
        format: Some("false".to_string()),
        format_check: None,
        lint_fix: None,
        lint: None,
        type_check: None,
        build: None,
        unit_test: None,
    };
    crate::workspace::store()
        .set_diagnostics(ws_name, &cmds, &crate::turso::now())
        .await
        .expect("set diagnostics");

    let ticket_id = make_ticket(
        board(),
        &ws,
        "Diagnostics Failure Test",
        TicketPhase::InDiagnostics,
    )
    .await;

    // NOTE: Do NOT claim the ticket beforehand — dispatch_diagnostics
    // calls claim_diagnostics internally as its first step.
    let ticket = expect_ticket(board(), &ticket_id).await;

    dispatch_diagnostics(Arc::new(ticket), ws).await;

    // Verify transition to ReadyForDevelopment with pipeline_reservation.
    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::ReadyForDevelopment,
        "diagnostics failure should bounce to ReadyForDevelopment",
    );
    assert!(
        ticket.pipeline_reservation,
        "bounced ticket should have pipeline_reservation = true",
    );

    // Verify a DIAGNOSTICS_ROLE comment was written with the failure marker.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert!(
        !comments.is_empty(),
        "should have written at least one comment",
    );
    let has_diag_failure = comments.iter().any(|c| {
        c.role == DIAGNOSTICS_ROLE
            && c.content.contains(DIAGNOSTICS_COMMENT_PREFIX)
            && c.content.contains(DIAGNOSTICS_FAILED_MARKER)
    });
    assert!(
        has_diag_failure,
        "should have a DIAGNOSTICS_ROLE comment with the failure marker",
    );

    // Keep dir alive until after the test completes.
    drop(dir);
}
