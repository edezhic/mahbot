use super::*;
use crate::prompt::{load_prompt, substitute};
use crate::util::test::make_ticket;
use crate::util::test::{
    create_test_workspace, expect_ticket, expect_ticket_phase, init_management_test_stores,
    init_test_stores,
};
use crate::workspace::test_ws_named;
use strum::IntoEnumIterator;

/// All non-TotalComments circuit breaker variants must have a `max_count` strictly
/// less than [`CircuitBreakerKind::TotalComments`]'s `max_count`.
///
/// ## Rationale
///
/// - **Sanitation breaker** (`max_count = 3`): must trip before the TotalComments
///   breaker (`max_count = 30`), otherwise a ticket could accumulate 30+
///   comments during repeated sanitation loops without tripping.
/// - **Diagnostics breaker** (`max_count = 4`): must also trip before the
///   TotalComments breaker. This is a conservative approximation — the TotalComments
///   breaker counts *all* comments (not just diagnostics), but guaranteeing
///   that diagnostics-only chatter cannot bypass the TotalComments breaker prevents
///   pathological ticket growth from repeated diagnostic cycles.

#[test]
fn all_non_total_comments_circuit_breakers_trip_before_total_comments() {
    let total_comments_max = CircuitBreakerKind::TotalComments.max_count();
    for kind in CircuitBreakerKind::iter() {
        if kind == CircuitBreakerKind::TotalComments {
            continue;
        }
        assert!(
            kind.max_count() < total_comments_max,
            "{kind:?}.max_count() ({}) must be less than TotalComments.max_count() ({total_comments_max})",
            kind.max_count(),
        );
    }
}

/// Verify the self-counting prevention invariant for non-terminal circuit
/// breakers (Sanitation, Diagnostics).
///
/// When a circuit breaker trips, it writes a comment to the ticket. On
/// re-evaluation, that trip comment must **not** be counted by the same
/// breaker variant — otherwise the breaker would trip again on every
/// subsequent poll cycle, creating an infinite loop.
///
/// ## How each variant prevents self-counting
///
/// | Variant | Role filter | Content filter |
/// |---------|-------------|----------------|
/// | **Sanitation** | `SYSTEM_ROLE` ≠ `SANITATION_ROLE` ✅ | content does **not** contain `sanitation_failed.md` ✅ |
/// | **Diagnostics** | `SYSTEM_ROLE` ≠ `DIAGNOSTICS_ROLE` ✅ | content does **not** contain `diagnostics_failed.md` ✅ |
/// | **TotalComments** | — | — (terminal `Failed` phase prevents re-evaluation) |
///
/// Both Sanitation and Diagnostics breakers now have role-based protection:
/// trip comments (written with `SYSTEM_ROLE`) are structurally excluded from
/// counting because the filter checks a different role. This is more robust
/// than content-substring exclusion alone.
///
/// This test verifies both the content-substring exclusion (the 80% case) and
/// — by feeding [`CircuitBreakerKind::should_trip`] with a [`TicketComment`]
/// constructed from the actual trip message — that the full filtering logic
/// would not count a trip comment (the 100% case).
#[test]
fn circuit_breaker_self_counting_prevention() {
    // ── Sanitation breaker: dual role + content exclusion ──
    {
        let msg = CircuitBreakerKind::Sanitation.trip_message(99, 3);

        // Full should_trip verification: a comment with SYSTEM_ROLE (the role actually
        // used by try_trip_circuit_breaker) and the trip message content should not be
        // counted (role mismatch: SYSTEM_ROLE ≠ SANITATION_ROLE).
        let trip_comment = TicketComment {
            role: SYSTEM_ROLE.to_owned(),
            content: msg,
            created_at: String::new(),
        };
        assert!(
            CircuitBreakerKind::Sanitation
                .should_trip(&[trip_comment])
                .is_none(),
            "Sanitation breaker must NOT count its own trip comment \
             (role mismatch: SYSTEM_ROLE != SANITATION_ROLE)",
        );
    }

    // ── Diagnostics breaker: dual role + content exclusion ──
    {
        let msg = CircuitBreakerKind::Diagnostics.trip_message(99, 4);

        // The trip message must NOT contain the diagnostics_failed.md marker string.
        let failed_marker = load_prompt("diagnostics_failed.md");
        assert!(
            !msg.contains(failed_marker.as_str()),
            "Diagnostics trip message must not contain the diagnostics_failed.md marker string \
             ({:?}), otherwise self-counting would occur on re-evaluation. Trip message: {msg:?}",
            failed_marker,
        );

        // Full should_trip verification: a comment with SYSTEM_ROLE (the role actually
        // used by try_trip_circuit_breaker) and the trip message content should not be
        // counted (role mismatch: SYSTEM_ROLE ≠ DIAGNOSTICS_ROLE).
        let trip_comment = TicketComment {
            role: SYSTEM_ROLE.to_owned(),
            content: msg,
            created_at: String::new(),
        };
        assert!(
            CircuitBreakerKind::Diagnostics
                .should_trip(&[trip_comment])
                .is_none(),
            "Diagnostics breaker must NOT count its own trip comment \
             (role mismatch: SYSTEM_ROLE != DIAGNOSTICS_ROLE)",
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
    // (CircuitBreakerKind::TotalComments.max_count() + 1 = 31 comments, enough to trip).
    for i in 0..=CircuitBreakerKind::TotalComments.max_count() {
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
        CircuitBreakerKind::TotalComments,
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
/// breaker variant.
///
/// For each variant:
/// - Adds `max_count` failures — verifies the breaker does NOT trip and phase
///   remains unchanged
/// - Adds one more failure — verifies the breaker trips, transitions to Failed,
///   and writes a trip comment with the "Circuit breaker" marker as a
///   [`SYSTEM_ROLE`] comment.
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
        BreakerCase {
            name: "TotalComments",
            kind: CircuitBreakerKind::TotalComments,
            source_phase: TicketPhase::InReview,
            log_label: "TotalComments",
            ws_suffix: "tc_breaker_test",
        },
    ];

    for case in &cases {
        let max_count = case.kind.max_count();
        let below_max = max_count; // Won't trip (count == max_count is still ≤ max_count)
        let trip_at = max_count + 1; // Will trip (count > max_count)

        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", case.ws_suffix),
            &format!("{} Breaker Test", case.log_label),
            case.source_phase,
        )
        .await;

        // Add max-count failures — should NOT trip.
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

        // Verify phase is unchanged after non-trip.
        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
            case.source_phase,
            "case {}: phase should remain {} after {} non-tripping failures (max: {})",
            case.name,
            case.source_phase,
            below_max,
            case.kind.max_count(),
        );

        // Add one more failure to reach the trip count.
        // Breaker trips when count > max_count.
        for _ in below_max..trip_at {
            add_breaker_failure(case.kind, &ticket_id).await;
        }

        // Re-fetch ticket (try_trip_circuit_breaker uses cached comments
        // when available — expect_ticket uses LoadComments::Yes, so the
        // cached path is exercised here).
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
/// For [`CircuitBreakerKind::Sanitation`], adds a [`SANITATION_ROLE`] comment
/// composed from `sanitation_circuit_breaker_comment.md` using `sanitation_failed.md`.
/// For [`CircuitBreakerKind::Diagnostics`], adds a [`DIAGNOSTICS_ROLE`] comment with
/// [`DIAGNOSTICS_COMMENT_PREFIX`] and `diagnostics_failed.md` with
/// `{{failed_at}}` = `"test_step"`.
async fn add_breaker_failure(kind: CircuitBreakerKind, ticket_id: &str) {
    let (role, comment) = match kind {
        CircuitBreakerKind::Sanitation => (
            SANITATION_ROLE,
            substitute(
                &load_prompt("sanitation_circuit_breaker_comment.md"),
                &[
                    (
                        "{{sanitation_failed_marker}}",
                        load_prompt("sanitation_failed.md").as_str(),
                    ),
                    ("{{count}}", "1"),
                ],
            ),
        ),
        CircuitBreakerKind::Diagnostics => (
            DIAGNOSTICS_ROLE,
            format!(
                "{DIAGNOSTICS_COMMENT_PREFIX}\n\n---\n{} test_step",
                load_prompt("diagnostics_failed.md"),
            ),
        ),
        CircuitBreakerKind::TotalComments => (
            "user",
            "Comment — circuit breaker boundary test".to_string(),
        ),
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

// ── Sanitation agent registration / comment routing wiring (mahbot-1035) ──

/// Sanitation dispatch must persist the SAME suffixed agent ID it registers
/// with the message router — the mahbot-1035 contract.
///
/// The board routes mid-work comments to the exact ID stored in
/// `assigned_to` (`route_comment_to_agents` → `try_route`).
/// [`BoardStore::claim_sanitation`] stores the unsuffixed base ID as a
/// placeholder; [`register_sanitation_agent`] must overwrite it with the
/// suffixed ID the run actually registers, otherwise comments are silently
/// dropped for the whole phase.
///
/// This test exercises the register+assign scaffolding directly (no LLM
/// involved — `dispatch_sanitation` itself would invoke a real provider) and
/// verifies end-to-end delivery through `add_comment` → message router.
#[tokio::test]
async fn sanitation_register_persists_registered_id() {
    init_management_test_stores().await;
    let ws = test_ws_named("/tmp/test_san_register", "ws_san_register");
    let ticket_id = make_ticket(board(), &ws, "San Register", TicketPhase::InSanitation).await;

    let (agent_id, mut rx) = register_sanitation_agent(&ticket_id).await;

    // The stored ID must be exactly the ID registered in the router — the
    // mismatch that broke comment routing.
    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.assigned_to.as_deref(),
        Some(agent_id.as_str()),
        "assigned_to must store the exact suffixed agent ID registered in the router"
    );

    // The ID must carry the run-unique suffix (fresh session per run — the
    // deliberate design from commit 728e79a; must not be dropped).
    let base = crate::session::ticket_agent_id(&ticket_id, crate::Role::Sanitation.as_str());
    assert!(
        agent_id.starts_with(&base) && agent_id.len() > base.len(),
        "sanitation agent ID must be suffixed for run isolation, got {agent_id}"
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

/// Pin the mahbot-1035 mismatch shape: an unsuffixed `assigned_to` (the
/// `claim_sanitation` placeholder) does NOT match the suffixed agent ID a
/// sanitation run registers — comments are silently dropped.
///
/// This is the regression the fix eliminates: [`register_sanitation_agent`]
/// overwrites the placeholder with the registered suffixed ID so routing
/// matches. The negative assertion guards against someone "simplifying" the
/// fix by dropping the suffix instead (which would regress per-run session
/// isolation).
#[tokio::test]
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
#[tokio::test]
async fn process_sanitation_verdict_cases() {
    /// All scenarios of [`process_sanitation_verdict`]. The two comment-marker
    /// fields use different types: [`Case::sanit_markers`] is `&[&str]`
    /// because a Sanitation role comment is *always* created; [`Case::sys_markers`]
    /// is `Option<Vec<&'static str>>` because a [`SANITATION_ROLE`] circuit-breaker
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

    let sanitation_failed_marker: &'static str = load_prompt("sanitation_failed.md").leak();
    let sys_markers_val: &'static [&'static str] =
        Box::leak(vec![sanitation_failed_marker].into_boxed_slice());

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
            sys_markers: Some(sys_markers_val),
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
                )
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
/// | Diagnostics failure | `false` | ReadyForDevelopment | true | `DIAGNOSTICS_COMMENT_PREFIX` + `diagnostics_failed.md` |
/// | Diagnostics pass | `true`, ... | DiagnosticsDone | false | `DIAGNOSTICS_COMMENT_PREFIX` + `diagnostics_passed.md` |
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

    let diagnostics_failed_marker: &'static str = load_prompt("diagnostics_failed.md").leak();
    let diagnostics_passed_marker: &'static str = load_prompt("diagnostics_passed.md").leak();

    const NO_DIAG_CMDS: &[&str] = &["No diagnostics commands are configured"];
    const DB_ERR: &[&str] = &["database error"];

    let fail_comment_contains: &'static [&'static str] =
        Box::leak(vec![DIAGNOSTICS_COMMENT_PREFIX, diagnostics_failed_marker].into_boxed_slice());
    let pass_comment_contains: &'static [&'static str] =
        Box::leak(vec![DIAGNOSTICS_COMMENT_PREFIX, diagnostics_passed_marker].into_boxed_slice());

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

// ── Diagnostics clippy gate safety net ──────────────────────────────

/// Create a git-inited cargo scratch crate under `dir` with the given
/// `main.rs` body, committed so `cargo clippy --fix` can run (cargo fix
/// requires a VCS unless `--allow-no-vcs` is passed, matching real
/// workspaces).
fn make_diagnostics_scratch_crate(dir: &tempfile::TempDir, main_body: &str) {
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"diaggate\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write scratch Cargo.toml");
    std::fs::create_dir_all(dir.path().join("src")).expect("create scratch src dir");
    std::fs::write(dir.path().join("src").join("main.rs"), main_body)
        .expect("write scratch main.rs");
    let git = |args: &[&str]| {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .status()
            .expect("run git");
        assert!(status.success(), "git {args:?} failed");
    };
    git(&["init", "-q"]);
    git(&["add", "-A"]);
    git(&[
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-qm",
        "init",
    ]);
}

/// The stored (broken) diagnostics commands: a plain lint-fix slot plus a
/// lint slot that is the gate-breaking compound (`--fix` + `-D warnings` in
/// one invocation, rust-clippy#17444). The runner must repair the lint slot
/// into the fix+gate chain and skip the now-redundant lint-fix slot (the
/// compound's fix pass is exactly the stored lint-fix).
fn gate_breaking_stored_commands() -> DiagnosticsCommands {
    DiagnosticsCommands {
        lint_fix: Some("cargo clippy --fix --allow-dirty".to_string()),
        lint: Some("cargo clippy --fix --allow-dirty -- -D warnings".to_string()),
        ..Default::default()
    }
}

/// The gate-breaking clippy compound (rust-clippy#17444 — `clippy --fix` +
/// `-D warnings` in one invocation silently disables the lint gate and exits 0
/// with unfixable warnings remaining) is repaired at execution time: the runner
/// rewrites it into the gate-preserving fix+gate chain and skips the
/// now-redundant separate lint-fix slot. A known-unfixable lint must still
/// fail the run — this is the acceptance test for "`-D warnings` enforced in
/// every diagnostics run".
///
/// The scratch crate carries `too_many_arguments` (9 args): it is
/// **warn-by-default** and **not auto-fixable** (verified on the pinned 1.97.1:
/// plain `cargo clippy` exits 0 with a warning; `cargo clippy -- -D warnings`
/// exits 101 with `could not compile`). The fix pass therefore exits 0, and the
/// gate pass fails **only because the repaired chain retains `-D warnings`** —
/// if a regression dropped the gate, the run would pass and this test would
/// fail. (The earlier `absurd_extreme_comparisons` crate was rejected for this
/// purpose: that lint is deny-by-default, so the gate was not isolated.)
///
/// The gate must fail for the right reason: `could not compile` is emitted only
/// when rustc breaks the build under the deny — a cargo usage error (the
/// fix-only-flag retention regression) or the fix pass (warnings only, exit 0)
/// never produce it — and the executed gate pass command (`cargo clippy --
/// -D warnings`) is printed verbatim in the comment, pinning the exact string
/// the runner executes.
#[tokio::test]
async fn diagnostics_repairs_gate_breaking_clippy_compound() {
    if !crate::git_commands::git_is_installed().await {
        eprintln!("git not installed — skipping git-dependent test");
        return;
    }
    init_management_test_stores().await;

    let dir = tempfile::tempdir().expect("create temp dir");
    let ws_path = dir.path().to_string_lossy().to_string();
    make_diagnostics_scratch_crate(
        &dir,
        "fn main() {\n    let _ = too_many(1, 2, 3, 4, 5, 6, 7, 8, 9);\n}\n\nfn too_many(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32) -> i32 {\n    a + b + c + d + e + f + g + h + i\n}\n",
    );

    let ws = create_test_workspace(&ws_path, "diag_gate_repair").await;

    let (comment, all_passed) =
        run_diagnostics_commands(&gate_breaking_stored_commands(), &ws).await;

    assert!(
        !all_passed,
        "the -D warnings gate must catch the warn-by-default lint — the run must fail"
    );
    assert!(
        comment.contains("&&"),
        "the broken compound should be repaired into a fix+gate chain, comment was:\n{comment}"
    );
    assert!(
        !comment.contains("lint-fix"),
        "the redundant lint-fix slot should be skipped when lint merges fix+gate, comment was:\n{comment}"
    );
    // The repaired gate pass command is printed verbatim in the comment's
    // `lint (…):` header — this pins the executed gate string itself, not just
    // its observable failure.
    assert!(
        comment.contains("cargo clippy -- -D warnings"),
        "the executed gate pass must carry `-D warnings`, comment was:\n{comment}"
    );
    assert!(
        comment.contains("could not compile"),
        "the -D warnings gate must break the build on the warn-by-default lint, comment was:\n{comment}"
    );
    let failed_marker = load_prompt("diagnostics_failed.md");
    assert!(
        comment.contains(&format!("{failed_marker} lint")),
        "failure attribution should name the lint slot, comment was:\n{comment}"
    );
}

/// Regression guard for the normalization fix: the repaired gate pass must not
/// carry fix-only flags (`--allow-dirty` / `--allow-no-vcs` / `--allow-staged`
/// / `--broken-code` are rejected by a plain `cargo clippy` invocation with a
/// usage error, exit 1, on the pinned 1.97.1 toolchain) — otherwise the
/// repaired chain fails on every run, even on lint-free code, turning the very
/// scenario the safety net exists to repair into an always-failing diagnostics
/// run.
#[tokio::test]
async fn diagnostics_repaired_chain_passes_on_clean_crate() {
    if !crate::git_commands::git_is_installed().await {
        eprintln!("git not installed — skipping git-dependent test");
        return;
    }
    init_management_test_stores().await;

    let dir = tempfile::tempdir().expect("create temp dir");
    let ws_path = dir.path().to_string_lossy().to_string();
    make_diagnostics_scratch_crate(&dir, "fn main() {\n    println!(\"clean\");\n}\n");

    let ws = create_test_workspace(&ws_path, "diag_gate_repair_clean").await;

    let (comment, all_passed) =
        run_diagnostics_commands(&gate_breaking_stored_commands(), &ws).await;

    assert!(
        all_passed,
        "the repaired chain must succeed on lint-free code, comment was:\n{comment}"
    );
    assert!(
        comment.contains("&&"),
        "the broken compound should still be repaired into a fix+gate chain, comment was:\n{comment}"
    );
    assert!(
        comment.contains(&load_prompt("diagnostics_passed.md")),
        "the run should end with the pass marker, comment was:\n{comment}"
    );
}

// ── dispatch_verifiers skip-review ──────────────────────────────

/// When a ticket is in InReview and the working tree has no unstaged changes,
/// dispatch_verifiers should skip spawning reviewers and transition directly
/// to Reviewed with a system comment explaining the skip.
#[tokio::test]
async fn dispatch_verifiers_skip_review_when_no_unstaged_changes() {
    if !crate::git_commands::git_is_installed().await {
        eprintln!("git not installed — skipping git-dependent test");
        return;
    }

    let (_dir, repo_path) = crate::util::test::init_temp_repo();
    let repo_str = repo_path.to_str().expect("temp path is valid UTF-8");

    let (ws, ticket_id) = setup_ticket(
        repo_str,
        "skip_review_test",
        "Skip Review Test",
        TicketPhase::InReview,
    )
    .await;

    let ticket = Arc::new(expect_ticket(board(), &ticket_id).await);

    dispatch_verifiers(ticket, ws, REVIEWER_VI).await;

    // Verify the ticket was transitioned to Reviewed.
    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::Reviewed,
        "Ticket with no unstaged changes should skip review and go directly to Reviewed"
    );

    // Verify a SYSTEM_ROLE comment was written explaining the skip.
    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get_comments");
    assert!(
        comments
            .iter()
            .any(|c| { c.role == SYSTEM_ROLE && c.content.contains("No unstaged changes") }),
        "Expected a SYSTEM_ROLE comment explaining the skip-review reason"
    );
}

// ── mahbot-1066 Amendment B: verdict raw-response comments ───────────────

/// Build a [`crate::retry::RetryExhausted`] with the given last-attempt raw text.
fn retry_exhausted_with_raw(last_raw: Option<String>) -> crate::retry::RetryExhausted {
    let rec = crate::retry::RetryFailureRecord::new_simple(
        1,
        crate::retry::FailureClass::Parse,
        &anyhow::anyhow!("parse failed"),
        100,
        None,
    );
    crate::retry::RetryExhausted::with_last_raw(
        vec![rec],
        crate::retry::FailureClass::Parse,
        last_raw,
    )
}

#[test]
fn parse_failed_comment_carries_raw_last_attempt_response() {
    let raw = r#"{"score": 9, "critique": "solid", "issues": []}"#;
    let pv = ParallelVerdict::ParseFailed(retry_exhausted_with_raw(Some(raw.to_string())));
    let comment = format_verdict_comment(&pv, "analyst_1", VerdictFilter::All)
        .expect("parse-failed must produce a comment");
    assert!(comment.contains("verdict extraction failed"), "{comment}");
    assert!(comment.contains("Raw agent response"), "{comment}");
    assert!(
        comment.contains(raw),
        "raw text must be in the comment: {comment}"
    );
    // The template's role attribution is preserved.
    assert!(comment.contains("analyst_1"), "{comment}");
}

#[test]
fn parse_failed_comment_sandwich_truncates_large_raw() {
    // 30_000-byte raw dump exceeds the 24_000-byte cap → sandwich marker.
    let big = format!("HEAD{}", "x".repeat(30_000));
    let pv = ParallelVerdict::ParseFailed(retry_exhausted_with_raw(Some(big)));
    let comment =
        format_verdict_comment(&pv, "reviewer_2", VerdictFilter::FailingOnly).expect("comment");
    assert!(
        comment.contains("bytes omitted at verdict response truncation"),
        "sandwich marker must appear: {}",
        &comment[..comment.len().min(300)]
    );
    assert!(comment.contains("HEAD"), "head must be preserved");
    assert!(
        comment.contains(&"x".repeat(8_000)),
        "tail must be preserved (shows where a mid-JSON cut landed)"
    );
    assert!(
        comment.len() < 26_000,
        "comment must be capped near the dump cap, got {}",
        comment.len()
    );
}

#[test]
fn parse_failed_comment_tool_call_final_attempt_note() {
    // Some(empty) — the final attempt was a tool call, no text.
    let pv = ParallelVerdict::ParseFailed(retry_exhausted_with_raw(Some(String::new())));
    let comment = format_verdict_comment(&pv, "qa_3", VerdictFilter::FailingOnly).expect("comment");
    assert!(
        comment
            .to_lowercase()
            .contains("final attempt was a tool call"),
        "{comment}"
    );
}

#[test]
fn parse_failed_comment_transport_failure_carries_trail() {
    // None — the final attempt died before producing text (transport).
    let mut failure = retry_exhausted_with_raw(None);
    failure.final_class = crate::retry::FailureClass::TruncatedEnvelope;
    let pv = ParallelVerdict::ParseFailed(failure);
    let comment = format_verdict_comment(&pv, "analyst_2", VerdictFilter::All).expect("comment");
    assert!(comment.contains("truncated_envelope"), "{comment}");
    assert!(comment.contains("1 attempt(s)"), "{comment}");
}

#[test]
fn no_response_keeps_existing_template() {
    let comment = format_verdict_comment(&no_verdict(), "analyst_1", VerdictFilter::All)
        .expect("no-response must produce a comment");
    assert!(
        comment.contains("failed to produce a response"),
        "{comment}"
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
