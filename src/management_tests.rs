use super::*;
use crate::prompt::{load_prompt, substitute};
use crate::util::test::make_ticket;
use crate::util::test::{
    create_test_workspace, expect_ticket, expect_ticket_phase, init_management_test_stores,
    init_test_stores,
};
use crate::workspace::test_ws_named;

/// The bounce-based breaker allows exactly [`MAX_BOUNCES`] bounces — the 11th
/// bounce fails the ticket. The comment-based breakers (Sanitation,
/// Diagnostics) must trip before the bounce budget so repeated agent
/// failures never ride on the bounce budget.
#[test]
fn bounce_breaker_max_is_ten_and_comment_breakers_trip_before() {
    let bounce_max = crate::joint_verdict::MAX_BOUNCES;
    assert_eq!(bounce_max, 10, "MAX_BOUNCES must be 10");
    for kind in [
        CircuitBreakerKind::Sanitation,
        CircuitBreakerKind::Diagnostics,
    ] {
        assert!(
            kind.max_count() < bounce_max,
            "{kind:?}.max_count() ({}) must be less than the bounce budget ({bounce_max})",
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
#[tokio::test]
async fn circuit_breaker_self_counting_prevention() {
    init_test_stores().await;

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
        let failed_marker = load_prompt("pipeline/diagnostics_failed.md");
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

    // Trip ticket A's circuit breaker: 4 cumulative sanitation failures
    // (max 3) — the drain behavior is breaker-agnostic, and the bounce
    // breaker has no pre-flight arm (it is enforced mid-round).
    for _ in 0..4 {
        add_breaker_failure(CircuitBreakerKind::Sanitation, &trip_id).await;
    }

    // Fetch ticket A and trip the circuit breaker.
    let ticket_a = expect_ticket(board(), &trip_id).await;

    let tripped = try_trip_circuit_breaker(
        &ticket_a,
        TicketPhase::ReadyForDevelopment,
        Some(CircuitBreakerKind::Sanitation),
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
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
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
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
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
            !try_trip_circuit_breaker(&ticket, case.source_phase, Some(case.kind), case.log_label,)
                .await,
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
            try_trip_circuit_breaker(&ticket, case.source_phase, Some(case.kind), case.log_label)
                .await;
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

/// Add a failure comment for circuit breaker testing, matching the
/// comment format used for the given breaker variant.
///
/// For [`CircuitBreakerKind::Sanitation`], adds a [`SANITATION_ROLE`] comment
/// composed from `sanitation_circuit_breaker_comment.md` using `sanitation_failed.md`.
/// For [`CircuitBreakerKind::Diagnostics`], adds a [`DIAGNOSTICS_ROLE`] comment with
/// `diagnostics_failed.md` and `{{failed_at}}` = `"test_step"`.
async fn add_breaker_failure(kind: CircuitBreakerKind, ticket_id: &str) {
    let (role, comment) = match kind {
        CircuitBreakerKind::Sanitation => (
            SANITATION_ROLE,
            substitute(
                &load_prompt("pipeline/sanitation_circuit_breaker_comment.md"),
                &[
                    (
                        "{{sanitation_failed_marker}}",
                        load_prompt("pipeline/sanitation_failed.md").as_str(),
                    ),
                    ("{{count}}", "1"),
                ],
            ),
        ),
        CircuitBreakerKind::Diagnostics => (
            DIAGNOSTICS_ROLE,
            format!(
                "---\n{} test_step",
                load_prompt("pipeline/diagnostics_failed.md")
            ),
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
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn install_synthesis_test_seams(
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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
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
        install_synthesis_test_seams(crate::util::test::FakeProvider::new()).await;

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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn verifier_finalization_on_moved_ticket_is_clean_skip() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new()).await;

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

/// A circuit breaker whose ticket was moved externally while the stage
/// finished is a silent no-op at the transition site too: the CAS guard miss
/// must NOT surface as a "may loop indefinitely" failure (that error fires
/// only on genuine write failures) and must NOT clobber the external mover's
/// assignment. The breaker still reports tripped so the caller aborts
/// dispatch.
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn circuit_breaker_on_moved_ticket_is_silent_noop() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new()).await;

    let ws = test_ws_named("/tmp/test", "cb_moved");
    let ticket_id = make_ticket(board(), &ws, "CB Moved", TicketPhase::InSanitation).await;
    for _ in 0..4 {
        add_breaker_failure(CircuitBreakerKind::Sanitation, &ticket_id).await;
    }

    // The Manager moves the ticket externally while the sanitation stage
    // finishes; the mover's claim must survive the breaker's Failed attempt.
    board()
        .transition_to(&ticket_id, None, TicketPhase::Planning, None)
        .await
        .expect("external move to Planning");
    board()
        .set_assigned_to_no_cancel(&ticket_id, Some("external-mover"))
        .await
        .expect("external mover claims the ticket");

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert!(
        try_trip_circuit_breaker(
            &ticket,
            TicketPhase::InSanitation,
            Some(CircuitBreakerKind::Sanitation),
            "Sanitation",
        )
        .await,
        "breaker still reports tripped — the caller must abort dispatch"
    );

    let ticket = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::Planning,
        "the moved ticket must be left untouched (no Failed transition)"
    );
    assert_eq!(
        ticket.assigned_to.as_deref(),
        Some("external-mover"),
        "guard miss must not clear the external mover's assignment"
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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn resume_verifier_round_replays_stored_outcomes() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new()).await;

    let ws = test_ws_named("/tmp/test", "vp_resume");
    let ticket_id = make_ticket(board(), &ws, "VP Resume", TicketPhase::InReview).await;
    let job_id = "vp_resume_job";
    let now = crate::turso::now();
    let conn = &crate::session::store().conn;

    // Job row + ticket_stage_jobs row exactly as a crashed dispatch leaves
    // them (stage=review, phase=in_review, round=1).
    conn.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
         created_at, updated_at) \
         VALUES (?1, 'ticket_stage', 'launched', '', ?2, 'reviewer', 0, ?3, ?3)",
        crate::turso::params![job_id, ws.name.clone(), now.clone()],
    )
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
        let agent_id = format!("ticket_{}_resume_{}_reviewer", ticket_id, i);
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
             VALUES (?1, ?2, 'verifier', ?3, 'done', ?4, '')",
            crate::turso::params![
                job_id,
                agent_id,
                i as i64,
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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn resume_engineer_round_continues_anchor_session() {
    init_management_test_stores().await;
    // Main LLM response + engineer-summary extraction (2 calls).
    let fake = crate::util::test::FakeProvider::new()
        .ok("implemented the resume path")
        .ok(r#"{"items": ["resumed the engineer session"]}"#);
    let (_lock, _policy_guard, _provider_guard) = install_synthesis_test_seams(fake).await;

    let ws = test_ws_named("/tmp/test", "eng_resume");
    let ticket_id = make_ticket(board(), &ws, "Eng Resume", TicketPhase::InDevelopment).await;
    let job_id = "eng_resume_job";
    let now = crate::turso::now();
    let conn = &crate::session::store().conn;
    let anchor_id = crate::jobs::engineer_anchor_id(&ticket_id);
    let task = "round 2 feedback: fix the tests";

    // Job + ticket_stage_jobs rows exactly as a crashed round-2 dispatch
    // leaves them (stage=engineer, phase=in_development, round=2).
    conn.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
         created_at, updated_at) \
         VALUES (?1, 'ticket_stage', 'launched', ?2, ?3, 'engineer', 0, ?4, ?4)",
        crate::turso::params![job_id, task, ws.name.clone(), now.clone()],
    )
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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn resume_sanitation_round_continues_session_and_passes() {
    init_management_test_stores().await;
    // Main LLM response + SanitationVerdict extraction (2 calls).
    let fake = crate::util::test::FakeProvider::new()
        .ok("inspected the workspace — no garbage files found")
        .ok(r#"{"pass": true, "garbage_files": [], "rationale": "workspace is clean"}"#);
    let (_lock, _policy_guard, _provider_guard) = install_synthesis_test_seams(fake).await;

    let ws = test_ws_named("/tmp/test", "san_resume");
    let ticket_id = make_ticket(board(), &ws, "San Resume", TicketPhase::InSanitation).await;
    let job_id = "san_resume_job";
    let now = crate::turso::now();
    let conn = &crate::session::store().conn;
    let agent_id = format!("ticket_{job_id}_sanitation");
    let task = "sanitation task";

    // Job + ticket_stage_jobs rows exactly as a crashed dispatch leaves them
    // (stage=sanitation, phase=in_sanitation, round=1).
    conn.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
         created_at, updated_at) \
         VALUES (?1, 'ticket_stage', 'launched', ?2, ?3, 'sanitation', 0, ?4, ?4)",
        crate::turso::params![job_id, task, ws.name.clone(), now.clone()],
    )
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
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn eleventh_bounce_fails_ticket() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new()).await;

    let ws = test_ws_named("/tmp/test", "eleventh_bounce");
    let ticket_id = make_ticket(board(), &ws, "Eleventh Bounce", TicketPhase::InReview).await;
    board()
        .conn
        .execute(
            "UPDATE tickets SET bounce_count = ?1 WHERE id = ?2",
            turso::params![crate::joint_verdict::MAX_BOUNCES as i64, ticket_id.as_str()],
        )
        .await
        .expect("set bounce_count to max");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let transitioned = process_verifier_verdicts(
        &ws,
        &ticket,
        &vec![pass_result(), fail_result(), pass_result()],
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
        crate::joint_verdict::MAX_BOUNCES as i64,
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
/// - A technical failure (all verifier agents failing) pauses the workspace.
/// - A circuit-breaker trip does NOT pause (it keeps its drain-to-Planning
///   handling) — the pause must be site-gated, not a generic Failed hook.
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases). Also serialized
/// with the drain-flag writers: `process_verifier_verdicts` and
/// `pause_workspace_on_failure` consult the process-global drain flag, which
/// would suppress the pause and the transition (project convention:
/// retry_tests_lock).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn technical_failure_pauses_workspace_but_circuit_breaker_does_not() {
    let _lock = crate::util::test::retry_tests_lock();
    init_management_test_stores().await;

    // ── Circuit-breaker trip must NOT pause the workspace ──────────────
    let ws_breaker = create_test_workspace("/tmp/pause_breaker_ws", "ws_pause_breaker").await;
    let breaker_id = make_ticket(
        board(),
        &ws_breaker,
        "Breaker Trip",
        TicketPhase::InSanitation,
    )
    .await;
    for _ in 0..4 {
        add_breaker_failure(CircuitBreakerKind::Sanitation, &breaker_id).await;
    }
    let ticket = expect_ticket(board(), &breaker_id).await;
    assert!(
        try_trip_circuit_breaker(
            &ticket,
            TicketPhase::InSanitation,
            Some(CircuitBreakerKind::Sanitation),
            "Sanitation",
        )
        .await,
        "breaker should trip"
    );
    let ws = crate::workspace::store()
        .get_by_name("ws_pause_breaker")
        .await
        .expect("get workspace")
        .expect("workspace exists");
    assert!(
        !ws.paused,
        "circuit-breaker trip must NOT pause the workspace"
    );

    // ── Verifier all-failed must pause the workspace ───────────────────
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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
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
        install_synthesis_test_seams(crate::util::test::FakeProvider::new()).await;

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
#[allow(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn analyst_round_fails_open_with_fallback_comment() {
    init_management_test_stores().await;
    // Script every synthesis attempt as a transport failure → exhaustion →
    // deterministic fallback.
    let fake = crate::util::test::FakeProvider::new()
        .err(crate::retry::FailureClass::Transport, "synthesis down")
        .err(crate::retry::FailureClass::Transport, "synthesis down")
        .err(crate::retry::FailureClass::Transport, "synthesis down");
    let (_lock, _policy_guard, _provider_guard) = install_synthesis_test_seams(fake).await;

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
    let agent_id = format!("ticket_test-job_sanitation");
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
        format!("ticket_test-job_sanitation"),
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

    let sanitation_failed_marker: &'static str =
        load_prompt("pipeline/sanitation_failed.md").leak();
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
/// | Diagnostics failure | `false` | ReadyForDevelopment | true | `diagnostics_failed.md` marker |
/// | Diagnostics pass | `true`, ... | DiagnosticsDone | false | `diagnostics_passed.md` marker |
/// | DB error (corrupt JSON) | N/A (corrupt) | DiagnosticsDone | false | "database error" |
///
/// Serialized with the reset_inflight_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[allow(clippy::too_many_lines)]
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

    init_management_test_stores().await;

    let diagnostics_failed_marker: &'static str =
        load_prompt("pipeline/diagnostics_failed.md").leak();
    let diagnostics_passed_marker: &'static str =
        load_prompt("pipeline/diagnostics_passed.md").leak();

    const NO_DIAG_CMDS: &[&str] = &["No diagnostics commands are configured"];
    const DB_ERR: &[&str] = &["database error"];

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
        Some("Agent exceeded maximum of 1000 LLM iterations"),
    );
    assert!(
        c.contains("Agent exceeded maximum of 1000 LLM iterations"),
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
#[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
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

// ── ticket_stage roster helpers ──────────────────────────────────────────
// These pin the agent-id / angle-cycling contract shared by
// `spawn_ticket_stage_round` (fresh dispatch) and `append_ticket_stage_slots`
// (analysis escalation). The helpers are the single home for both rules — if
// the shape ever changes, these tests are the first to notice.

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
