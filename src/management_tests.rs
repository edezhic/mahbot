use super::*;
use crate::util::test::make_ticket;
use crate::util::test::{
    create_test_workspace, expect_ticket, expect_ticket_phase, init_management_test_stores,
    init_test_stores,
};
use crate::workspace::test_ws_named;
use strum::IntoEnumIterator;

/// All non-General circuit breaker variants must have a `max_count` strictly
/// less than [`CircuitBreakerKind::General`]'s `max_count`.
///
/// ## Rationale
///
/// - **Sanitation breaker** (`max_count = 3`): must trip before the general
///   breaker (`max_count = 30`), otherwise a ticket could accumulate 30+
///   comments during repeated sanitation loops without tripping.
/// - **Diagnostics breaker** (`max_count = 4`): must also trip before the
///   general breaker. This is a conservative approximation — the general
///   breaker counts *all* comments (not just diagnostics), but guaranteeing
///   that diagnostics-only chatter cannot bypass the general breaker prevents
///   pathological ticket growth from repeated diagnostic cycles.

#[test]
fn all_non_general_circuit_breakers_trip_before_general() {
    let general = CircuitBreakerKind::General.max_count();
    for kind in CircuitBreakerKind::iter() {
        if kind == CircuitBreakerKind::General {
            continue;
        }
        assert!(
            kind.max_count() < general,
            "{kind:?}.max_count() ({}) must be less than General.max_count() ({general})",
            kind.max_count(),
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
    // (CircuitBreakerKind::General.max_count() + 1 = 31 comments, enough to trip).
    for i in 0..=CircuitBreakerKind::General.max_count() {
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

    // After the breaker trips, drain siblings so the Manager can triage
    // without new tickets auto-starting.
    drain_ready_for_development_siblings(&ticket_a).await;

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

// ── try_trip_circuit_breaker — failure counting ──────────────────

/// Verify that circuit breaker counting logic works correctly for each
/// non-General breaker variant.
///
/// For each variant:
/// - Adds below-max-count failures — verifies the breaker does NOT trip
/// - Adds more failures to reach the trip count — verifies the breaker
///   trips, transitions to Failed, and writes a trip comment with the
///   "Circuit breaker" marker as a SYSTEM_ROLE comment.
#[tokio::test]
async fn breaker_counts_failures() {
    struct BreakerCase {
        name: &'static str,
        kind: CircuitBreakerKind,
        source_phase: TicketPhase,
        log_label: &'static str,
        ws_suffix: &'static str,
    }

    init_management_test_stores().await;

    let cases = [
        BreakerCase {
            name: "Sanitation",
            kind: CircuitBreakerKind::Sanitation,
            source_phase: TicketPhase::InSanitation,
            log_label: "Sanitation",
            ws_suffix: "san_breaker_test",
        },
        BreakerCase {
            name: "Diagnostics",
            kind: CircuitBreakerKind::Diagnostics,
            source_phase: TicketPhase::InDiagnostics,
            log_label: "Diagnostics",
            ws_suffix: "diag_breaker_test",
        },
    ];

    for case in &cases {
        let max_count = case.kind.max_count();
        let below_max = max_count - 1; // Won't trip
        let trip_at = max_count + 1; // Will trip (count > max_count)

        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", case.ws_suffix),
            &format!("{} Breaker Test", case.log_label),
            case.source_phase,
        )
        .await;

        // Add below-max-count failures.
        for _ in 0..below_max {
            add_breaker_failure(case.kind, &ticket_id).await;
        }

        let ticket = expect_ticket(board(), &ticket_id).await;

        assert!(
            !try_trip_circuit_breaker(&ticket, case.source_phase, case.kind, case.log_label,).await,
            "case {}: should NOT trip with {} failures (max: {})",
            case.name,
            below_max,
            case.kind.max_count(),
        );

        // Add more failures to reach the trip count.
        // Breaker trips when count > max_count.
        for _ in below_max..trip_at {
            add_breaker_failure(case.kind, &ticket_id).await;
        }

        // Re-fetch ticket (comments are refetched internally by
        // try_trip_circuit_breaker, so we just need the ID).
        let ticket = expect_ticket(board(), &ticket_id).await;

        let tripped =
            try_trip_circuit_breaker(&ticket, case.source_phase, case.kind, case.log_label).await;
        assert!(
            tripped,
            "case {}: should trip with {} failures (max: {}, {} > {})",
            case.name,
            trip_at,
            case.kind.max_count(),
            trip_at,
            case.kind.max_count(),
        );

        // Verify the ticket is now Failed
        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
            TicketPhase::Failed,
            "case {}: circuit breaker should transition to Failed",
            case.name,
        );

        // Verify the trip comment was written correctly:
        // must be a SYSTEM_ROLE comment containing "circuit breaker"
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        let has_breaker_comment = comments
            .iter()
            .any(|c| c.role == SYSTEM_ROLE && c.content.to_lowercase().contains("circuit breaker"));
        assert!(
            has_breaker_comment,
            "case {}: should have a SYSTEM_ROLE comment with the circuit breaker message \
             (containing 'circuit breaker')",
            case.name,
        );
    }
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

/// Add a failure comment for circuit breaker testing, matching the
/// comment format used for the given breaker variant.
///
/// For [`CircuitBreakerKind::Sanitation`], adds a [`SYSTEM_ROLE`] comment
/// with [`SANITATION_FAILED_MARKER`]. For [`CircuitBreakerKind::Diagnostics`],
/// adds a [`DIAGNOSTICS_ROLE`] comment with [`DIAGNOSTICS_COMMENT_PREFIX`]
/// and [`DIAGNOSTICS_FAILED_MARKER`].
async fn add_breaker_failure(kind: CircuitBreakerKind, ticket_id: &str) {
    let (role, comment) = match kind {
        CircuitBreakerKind::Sanitation => (
            SYSTEM_ROLE,
            format!("{SANITATION_FAILED_MARKER} — garbage files: 1"),
        ),
        CircuitBreakerKind::Diagnostics => (
            DIAGNOSTICS_ROLE,
            format!("{DIAGNOSTICS_COMMENT_PREFIX}\n\n---\n{DIAGNOSTICS_FAILED_MARKER} test_step"),
        ),
        CircuitBreakerKind::General => {
            unreachable!("General breaker not used in failure-counting tests")
        }
    };
    let _ = board().add_comment(ticket_id, role, &comment).await;
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

/// Verify the circuit breaker trips at the max_count boundary:
/// - `> CircuitBreakerKind::General.max_count()` comments → trips (ticket → Failed)
/// - `= CircuitBreakerKind::General.max_count()` comments → does NOT trip
///
/// When the breaker trips, also verifies the trip comment contains the
/// "circuit breaker" marker as produced by [`CircuitBreakerKind::should_trip`].
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
            name: "> max_count trips",
            ws_suffix: "cb_max_count",
            title: "CB Max Count",
            comment_count: CircuitBreakerKind::General.max_count() + 1,
            expected_trip: true,
            expected_phase: TicketPhase::Failed,
        },
        Case {
            name: "= max_count does not trip",
            ws_suffix: "cb_no_trip",
            title: "CB No Trip",
            comment_count: CircuitBreakerKind::General.max_count(),
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
        // circuit breaker marker as produced by `CircuitBreakerKind::should_trip`.
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
/// - pass=true, clean → SanitationPassed, no system comment
/// - pass=false, garbage → ReadyForDevelopment with pipeline reservation and system comment
/// - pass=true, reviewed files → SanitationPassed with "(files reviewed)" suffix, no system comment
#[tokio::test]
async fn process_sanitation_verdict_cases() {
    /// All scenarios of [`process_sanitation_verdict`]. The two comment-marker
    /// fields use different types by design: [`Case::sanit_markers`] is `&[&str]`
    /// because a Sanitation role comment is *always* created; [`Case::sys_markers`]
    /// is `Option<&[&str]>` because a SYSTEM circuit-breaker comment is
    /// *conditional* (only appears on `pass=false`).
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        verdict: crate::SanitationVerdict,
        expected_phase: TicketPhase,
        expected_pipeline_reservation: bool,
        /// Substrings required in a Sanitation role comment (empty = just exists).
        sanit_markers: &'static [&'static str],
        /// Substrings required in a SYSTEM role comment. `None` = no system comment.
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
            sys_markers: Some(&[SANITATION_FAILED_MARKER]),
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

        // System role check
        match case.sys_markers {
            Some(markers) => assert!(
                comments.iter().any(
                    |c| c.role == SYSTEM_ROLE && markers.iter().all(|&m| c.content.contains(m))
                ),
                "case {}: expected SYSTEM comment matching {:?}",
                case.name,
                markers,
            ),
            None => assert!(
                !comments.iter().any(|c| c.role == SYSTEM_ROLE),
                "case {}: expected no SYSTEM comment",
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
/// | Diagnostics failure | `false` | ReadyForDevelopment | true | DIAGNOSTICS_COMMENT_PREFIX + DIAGNOSTICS_FAILED_MARKER |
/// | Diagnostics pass | `true`, ... | DiagnosticsDone | false | DIAGNOSTICS_COMMENT_PREFIX + DIAGNOSTICS_PASSED_MARKER |
/// | DB error (corrupt JSON) | N/A (corrupt) | DiagnosticsDone | false | "database error" |
#[allow(clippy::too_many_lines)]
#[tokio::test]
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

    init_management_test_stores().await;

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
            expected_comment_contains: &["No diagnostics commands are configured"],
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
            expected_comment_contains: &[DIAGNOSTICS_COMMENT_PREFIX, DIAGNOSTICS_FAILED_MARKER],
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
            expected_comment_contains: &[DIAGNOSTICS_COMMENT_PREFIX, DIAGNOSTICS_PASSED_MARKER],
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
            expected_comment_contains: &["database error"],
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
                .set_diagnostics(case.ws_suffix, cmds, &crate::turso::now())
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
