use super::*;
use crate::board::TicketComment;
use crate::util::test::make_ticket;
use crate::util::test::{
    JobRowBuilder, create_test_workspace, expect_ticket, expect_ticket_phase,
    init_management_test_stores,
};
use crate::workspace::test_ws_named;

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
/// [`make_ticket`].
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

/// Verify the Buffer → Notify + drain sequence across two InQa tickets
/// via `transition_ticket_to_done`: the first one buffers, the last one
/// notifies and drains the buffer.
///
/// Serialized with the reset_analysis_tickets tests (shared global board)
/// and holds retry_tests_lock: the Notify path routes a Manager notification
/// whose consumer loop runs a Manager agent that reads the process-global
/// provider (project convention: retry_tests_lock).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes the process-global seams
async fn transition_ticket_to_done_buffer_and_notify() {
    let _lock = crate::util::test::retry_tests_lock();
    let ws = setup_db_workspace("drains_buffer").await;

    // Two InQa tickets in the same workspace
    let first_id = make_ticket(board(), &ws, "Ticket A", TicketPhase::InQa).await;
    let second_id = make_ticket(board(), &ws, "Ticket B", TicketPhase::InQa).await;

    let ticket_a = expect_ticket(board(), &first_id).await;

    // Transition ticket A — ticket B is still InQa (active), so Buffer
    transition_ticket_to_done(
        &ticket_a,
        TicketPhase::InQa,
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
        "After first InQa → Done with other active tickets: \
             should have buffered the notification (got empty buffer)",
    );

    // Transition ticket B — no more active tickets, should Notify and drain
    let ticket_b = expect_ticket(board(), &second_id).await;
    transition_ticket_to_done(
        &ticket_b,
        TicketPhase::InQa,
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
/// - Any failed (including an all-failed round) → unified bounce to
///   InDevelopment, a single joint comment (role = stage name), and a bumped
///   bounce counter
/// - The bounce circuit breaker → Failed on exhaustion
/// - All passed (Reviewer) → InQa with a joint comment
/// - All passed (QA) → InSanitation with a joint comment
///
/// Serialized with the reset_analysis_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).

/// The stage-finalization choke point treats a phase-guard miss (ticket moved
/// externally while the stage was finishing) as a first-class, expected
/// outcome: the whole round is rolled back silently — nothing is written, no
/// bounce is counted — and the job's launched roster rows are NOT cleared by
/// the finalizer (the external mover already handled the ticket; a finalizer clear
/// would clobber a fresh claim by the new phase).
///
/// The round uses a mixed verdict (pass/fail/pass → any_failed), so the
/// target is InDevelopment and the closure increments the bounce
/// counter: the `bounce_count == 0` assertion genuinely exercises the
/// rollback of an in-transaction write that WOULD have committed had the
/// guard applied.
///
/// Regression guard for the structural fix: the guard miss is a silent skip
/// (the silent-rollback path never reaches the warn arms), not an error —
/// this test asserts the observable side effects (phase untouched, bounce
/// unchanged, active-agent registration preserved, no comment written).
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
    // finishes. The mover's transition must survive the finalizer.
    board()
        .transition_to(&ticket_id, None, TicketPhase::Planning)
        .await
        .expect("external move to Planning");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let transitioned = process_verifier_verdicts(
        &ws,
        &ticket,
        // Mixed verdict → any_failed → bounce to development: the
        // closure writes a joint comment AND increments the bounce counter,
        // so the guard miss must roll both back.
        &[pass_result(), fail_result(), pass_result()],
        REVIEWER_VI,
        "test_job",
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
        .transition_to(&ticket_id, None, TicketPhase::Planning)
        .await
        .unwrap();
    let slots: Vec<AgentSlot> = (0..2)
        .map(|i| AgentSlot {
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
/// Serialized with the reset_analysis_tickets tests (shared global board + the
/// process-global drain flag).
#[tokio::test]
#[serial_test::serial(reset_inflight, provider)] // process-global board + fake provider (providers::PROVIDER)
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
/// Serialized with the reset_analysis_tickets tests (shared global board — a
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
        "test_job",
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

// ── Claim gate: automatic claims blocked while paused or not Ready ──

/// The claim gate ([`blocks_claim`]) must block the automatic pickup of
/// *new* work (BacklogAnalysis, EngineerDevelopment) when the workspace is
/// paused **or** not Ready, and freeze the entire occupied implementation when the
/// workspace is paused. Two gates:
///
/// * **Pause gate** — a paused Ready workspace blocks ALL claims (the implementation
///   is frozen; unpause re-arms the poll) — pause is a real freeze, not just
///   a new-work gate.
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
        // Pause gate: a paused Ready workspace freezes every claim.
        let paused = Workspace {
            status: WorkspaceStatus::Ready,
            paused: true,
            ..Default::default()
        };
        assert!(
            blocks_claim(&paused, phase),
            "pause gate mismatch for {} ({} → {}) — a paused workspace must freeze the implementation",
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
/// - The spawned dispatches may persist `ticket_jobs`/`agents` rows
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
/// Serialized with the reset_analysis_tickets tests (shared global board — a
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
/// Serialized with the reset_analysis_tickets tests (shared global board — a
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

// ── process_sanitation_verdict — verdict processing ──────────────────

/// Verify [`process_sanitation_verdict`] across all scenarios:
/// - pass=true, clean → Done, no marker comment
/// - pass=false, garbage → InDevelopment (unified bounce), no marker comment
/// - pass=true, reviewed files → Done with "(files reviewed)" suffix, no marker comment
///
/// Serialized with the reset_analysis_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn process_sanitation_verdict_cases() {
    /// All scenarios of [`process_sanitation_verdict`]. [`Case::sanit_markers`]
    /// is `&[&str]` because a Sanitation role comment is *always* created.
    struct Case {
        name: &'static str,
        ws_suffix: &'static str,
        verdict: crate::SanitationVerdict,
        expected_phase: TicketPhase,
        /// Substrings required in a Sanitation role comment (empty = just exists).
        sanit_markers: &'static [&'static str],
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
            name: "pass=true → Done",
            ws_suffix: "sp",
            verdict: clean,
            expected_phase: TicketPhase::Done,
            sanit_markers: &[],
        },
        Case {
            name: "pass=false → InDevelopment (unified bounce)",
            ws_suffix: "sf",
            verdict: garbage,
            expected_phase: TicketPhase::InDevelopment,
            sanit_markers: &["node_modules/"],
        },
        Case {
            name: "pass=true with reviewed files → Done (files reviewed)",
            ws_suffix: "sp_r",
            verdict: reviewed,
            expected_phase: TicketPhase::Done,
            sanit_markers: &["(files reviewed)"],
        },
    ];

    for case in &cases {
        let ws = test_ws_named("/tmp/test", case.ws_suffix);
        let id = make_ticket(board(), &ws, case.name, TicketPhase::InSanitation).await;
        // The sanitation round runs on the implementation job (stage='sanitation');
        // create a real job row so the id handed to the verdict processor corresponds
        // to a real job rather than a phantom that only produces no-op synced writes.
        let job_id = format!("job_san_{}", case.ws_suffix);
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            &job_id,
            "task",
            &ws.name,
            "",
            "",
            Role::Engineer,
            &[],
            &crate::jobs::SpawnChild::TicketJob {
                ticket_id: id.clone(),
                is_implementation: true,
            },
        )
        .await
        .expect("create implementation job");
        let ticket = expect_ticket(board(), &id).await;
        process_sanitation_verdict(&ticket, &job_id, case.verdict.clone(), &ws).await;

        let phase = expect_ticket_phase(board(), &id).await;
        assert_eq!(
            phase, case.expected_phase,
            "case {}: expected phase {:?}, got {:?}",
            case.name, case.expected_phase, phase,
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

        // Marker role comment check: no SANITATION_ROLE comment is written.
        assert!(
            !comments.iter().any(|c| c.role == SANITATION_ROLE),
            "case {}: expected no SANITATION_ROLE comment",
            case.name,
        );
    }
}

// ── dispatch_verifiers skip-review ──────────────────────────────

/// When the current content is identical to the ticket's recorded reviewed
/// base (same HEAD, same index tree, clean porcelain), the reviewer pass may
/// legitimately be skipped — this is the comment-only-round case.
///
/// Serialized with the reset_analysis_tickets tests (shared global board — a
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

    dispatch_verifiers(ticket, ws, REVIEWER_VI, false).await;

    let phase = expect_ticket_phase(board(), &ticket_id).await;
    assert_eq!(
        phase,
        TicketPhase::InQa,
        "Content identical to the reviewed base should skip review and go directly to InQa"
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
#[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
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

// ── Engineer failure paths (cancel, hard-failure, drain) ────────────────

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

/// A genuine hard "Agent failed" outcome must NOT fail the ticket: it pauses
/// the workspace and freezes the implementation in place (the ticket stays in
/// InDevelopment), and records the concrete error as a SYSTEM-role comment so
/// the Manager notification and the resumed engineer's feedback surface it.

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

    finalize_engineer_stage(&ticket, &agent, None, "job_eng_cancel", &ws, false).await;

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

/// A genuine engineer hard failure (non-cancelled) pauses the workspace and
/// freezes the implementation in place: the ticket stays in InDevelopment (no bounce,
/// no bounce-budget consumption), the workspace is paused and `jobs.status`
/// stays 'launched', the implementation's launched roster rows are cleared, and a
/// SYSTEM-role comment carrying the failure reason is written (so the Manager
/// notification and the resumed engineer's feedback surface it).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: install_synthesis_test_seams holds the lock
async fn engineer_hard_failure_pauses_workspace_and_freezes_implementation() {
    init_management_test_stores().await;
    let (_lock, _policy_guard, _provider_guard) =
        install_synthesis_test_seams(crate::util::test::FakeProvider::new());

    let ws = create_test_workspace("/tmp/eng_hard_ws", "ws_eng_hard").await;
    let ticket_id = make_ticket(board(), &ws, "Eng Hard", TicketPhase::InDevelopment).await;
    let job_id = "job_eng_hard";
    let conn = &crate::session::store().conn;
    crate::jobs::spawn_job(
        conn,
        job_id,
        "task",
        &ws.name,
        "",
        "",
        Role::Engineer,
        &[],
        &crate::jobs::SpawnChild::TicketJob {
            ticket_id: ticket_id.clone(),
            is_implementation: true,
        },
    )
    .await
    .expect("create implementation job");

    let ticket = expect_ticket(board(), &ticket_id).await;
    let mut agent = engineer_finalize_test_agent(&ws, &ticket, "hard");
    agent.failure = Some("boom".to_string());

    finalize_engineer_stage(&ticket, &agent, None, job_id, &ws, false).await;

    let t = expect_ticket(board(), &ticket_id).await;
    assert_eq!(
        t.phase,
        TicketPhase::InDevelopment,
        "an engineer hard failure must freeze the implementation in place, NOT fail the ticket"
    );
    assert_eq!(
        t.bounce_count, 0,
        "an engineer hard failure must not consume the bounce budget"
    );

    let ws_after = crate::workspace::store()
        .get_by_name("ws_eng_hard")
        .await
        .expect("query workspace")
        .expect("workspace exists");
    assert!(
        ws_after.paused,
        "an engineer hard failure must pause the workspace"
    );

    let status: Option<String> = conn
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
        "the implementation job must stay 'launched' (frozen, not done) after the hard failure"
    );

    let active: i64 = conn
        .query_optional(
            "SELECT COUNT(*) FROM agents WHERE job_id = ?1 AND status = 'launched'",
            crate::turso::params![job_id],
            |row| row.get(0),
        )
        .await
        .expect("query active agents")
        .expect("active-agent count row exists");
    assert_eq!(
        active, 0,
        "the engineer's running roster row must be cleared"
    );

    let comments = board()
        .get_comments(&ticket_id)
        .await
        .expect("get comments");
    let last = comments.last().expect("failure comment written");
    assert_eq!(
        last.role, SYSTEM_ROLE,
        "the hard-failure comment is SYSTEM-role"
    );
    assert!(
        last.content.contains("boom"),
        "the failure reason must be recorded: {}",
        last.content
    );
}

/// The InDevelopment re-dispatch lane must feed outstanding feedback to the
/// re-dispatched engineer. The poll loads the ticket with its comments
/// (via `get_ticket`, `LoadComments::Yes`), so [`engineer_work_message`] sees the
/// post-Engineer validation/rejecter comments and selects the bounce-feedback
/// prompt when one exists, or the plain implement prompt otherwise.
#[tokio::test]
#[serial_test::serial(reset_inflight)]
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn in_development_redispatch_lane_feeds_feedback() {
    init_management_test_stores().await;
    let _lock = crate::util::test::retry_tests_lock();

    let ws = create_test_workspace("/tmp/eng_redispatch_ws", "ws_eng_redispatch").await;
    let ticket_id = make_ticket(board(), &ws, "Eng Redispatch", TicketPhase::InDevelopment).await;
    let job_id = "job_eng_redispatch";
    let conn = &crate::session::store().conn;
    crate::jobs::spawn_job(
        conn,
        job_id,
        "task",
        &ws.name,
        "",
        "",
        Role::Engineer,
        &[],
        &crate::jobs::SpawnChild::TicketJob {
            ticket_id: ticket_id.clone(),
            is_implementation: true,
        },
    )
    .await
    .expect("create implementation job");

    // An Engineer comment, then a post-Engineer validation failure comment.
    board()
        .add_comment(
            &ticket_id,
            Role::Engineer.as_str(),
            "round 1 implementation",
        )
        .await
        .expect("add engineer comment");
    board()
        .add_comment(
            &ticket_id,
            Role::Reviewer.as_str(),
            "please fix the type error",
        )
        .await
        .expect("add reviewer comment");

    // The lane's re-fetch uses get_ticket (LoadComments::Yes), so comments are
    // present and the message carries the post-Engineer feedback.
    let ticket = expect_ticket(board(), &ticket_id).await;
    let message = engineer_work_message(&ticket);
    let expected = substitute(
        &load_prompt("pipeline/bounce_feedback.md"),
        &[("{{feedback}}", "please fix the type error")],
    );
    assert_eq!(
        message, expected,
        "the re-dispatched engineer must receive the post-Engineer feedback"
    );
    assert!(
        message.contains("please fix the type error"),
        "the feedback comment text must be embedded in the prompt"
    );

    // With only a pre-Engineer comment, the feedback set is empty → implement.
    let second_id = make_ticket(board(), &ws, "Eng No Feedback", TicketPhase::InDevelopment).await;
    board()
        .add_comment(&second_id, Role::Analyst.as_str(), "pre-engineer note")
        .await
        .expect("add analyst comment");
    board()
        .add_comment(
            &second_id,
            Role::Engineer.as_str(),
            "round 1 implementation",
        )
        .await
        .expect("add engineer comment");
    let ticket = expect_ticket(board(), &second_id).await;
    let message = engineer_work_message(&ticket);
    assert_eq!(
        message,
        load_prompt("implement.md"),
        "no post-Engineer feedback → implement.md"
    );
}

/// The direct-bounce auto-rework path must re-fetch the ticket fresh AFTER the
/// validation failure comment is written, so the re-dispatched engineer receives
/// the failure feedback (bounce_feedback) rather than the bare implement prompt.
/// A stale in-memory clone would drop the just-written comment and regress the
/// "a resumed engineer gets the outstanding feedback" invariant.
///
/// This drives the REAL `bounce_to_development` → spawned `dispatch_engineer`
/// path and observes the first LLM request the re-dispatched engineer actually
/// sends to the provider (captured by the `FakeProvider`). That request's user
/// message IS `engineer_work_message(&ticket)` — the exact prompt handed to the
/// engineer. If the re-fetch were reverted (a stale clone handed to
/// `dispatch_engineer`), the prompt would be the bare `implement.md` and this
/// assertion fails; it does NOT independently re-fetch the ticket.
///
/// The first provider request is asserted because a successful engineer round
/// cascades into diagnostics/review, whose agents make further requests and
/// overwrite `jobs.task` — but `request_messages[0]` is the immutable original
/// re-dispatch prompt.
#[tokio::test]
#[serial_test::serial(reset_inflight, provider)] // process-global board + fake provider (providers::PROVIDER)
#[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
async fn bounce_to_development_feeds_feedback_to_redispatch() {
    use crate::util::test::{FakeProvider, install_fake_provider, install_test_retry_policy};

    init_management_test_stores().await;
    let _lock = crate::util::test::retry_tests_lock();
    // The bounce spawns a real `dispatch_engineer` which drives a provider round;
    // a fake provider records the prompt (and keeps the detached task from
    // touching a real endpoint at teardown).
    let _policy_guard = install_test_retry_policy(crate::retry::tiny_test_policy());
    let fake = std::sync::Arc::new(FakeProvider::new().ok("implemented"));
    let _provider = install_fake_provider(fake.clone());

    let ws = create_test_workspace("/tmp/eng_bounce_feedback_ws", "ws_eng_bounce_feedback").await;
    let ticket_id = make_ticket(
        board(),
        &ws,
        "Eng Bounce Feedback",
        TicketPhase::InDiagnostics,
    )
    .await;
    let job_id = "job_eng_bounce_feedback";
    let conn = &crate::session::store().conn;
    crate::jobs::spawn_job(
        conn,
        job_id,
        "task",
        &ws.name,
        "",
        "",
        Role::Engineer,
        &[],
        &crate::jobs::SpawnChild::TicketJob {
            ticket_id: ticket_id.clone(),
            is_implementation: true,
        },
    )
    .await
    .expect("create implementation job");

    // The bounce writes the validation failure comment and re-dispatches.
    let stale = expect_ticket(board(), &ticket_id).await;
    let outcome = bounce_to_development(
        &stale,
        TicketPhase::InDiagnostics,
        "Diagnostics",
        /* drains_siblings */ true,
        DIAGNOSTICS_ROLE,
        "failed: type error",
        job_id,
        &ws,
    )
    .await;
    assert!(matches!(outcome, FinalizeOutcome::Applied));

    // Wait for the spawned dispatch_engineer to make its first LLM request.
    let mut messages = Vec::new();
    for _ in 0..50 {
        messages = fake.request_messages.lock().unwrap().clone();
        if !messages.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        !messages.is_empty(),
        "the re-dispatched engineer never reached the provider"
    );

    let first = &messages[0];
    assert!(
        first.contains("failed: type error"),
        "re-dispatched engineer's first prompt must carry the just-written failure feedback; got: {first}"
    );
    assert!(
        !first.contains(&load_prompt("implement.md")),
        "a bounce re-dispatch must select the feedback prompt, not the bare implement"
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
        "ticket_implementation",
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
    finalize_engineer_stage(&ticket, &agent, None, job_id, &ws, false).await;
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
/// the caller's initial [`stage_drain_cut`]) — must bail without any
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
        "ticket_implementation",
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
    let bailed = handle_engineer_failure(&ticket, &agent, "job_midtail", &ws, false).await;
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

// ── ticket_analysis roster helpers ──────────────────────────────────────────
// These pin the agent-id / angle-cycling contract shared by
// `spawn_ticket_analysis_round` (fresh dispatch) and `append_ticket_analysis_slots`
// (analysis escalation). The fresh-dispatch path funnels its job+child-row
// spawn tail through `spawn_job`. The helpers are the single
// home for both rules — if the shape ever changes, these tests are the first
// to notice.

/// The agent-id helper must produce the exact documented shape
/// `ticket_{ticket_id}_{idx}_{suffix}_{role}` for both dispatch paths.
#[test]
fn agent_id_format() {
    assert_eq!(
        agent_id("t-42", 0, "abc123", Role::Analyst),
        "ticket_t-42_0_abc123_analyst",
        "base-round slot 0"
    );
    assert_eq!(
        agent_id("t-42", 2, "abc123", Role::Analyst),
        "ticket_t-42_2_abc123_analyst",
        "base-round slot 2"
    );
    // Escalation continues at the roster length (3, 4) with a FRESH suffix.
    assert_eq!(
        agent_id("t-42", 3, "def456", Role::Analyst),
        "ticket_t-42_3_def456_analyst",
        "escalation slot 3"
    );
    assert_eq!(
        agent_id("t-42", 4, "def456", Role::Analyst),
        "ticket_t-42_4_def456_analyst",
        "escalation slot 4"
    );
    // Role string is the canonical lowercase `as_str()` (role LAST).
    assert_eq!(
        agent_id("t-7", 0, "xyz789", Role::Reviewer),
        "ticket_t-7_0_xyz789_reviewer"
    );
    assert_eq!(
        agent_id("t-7", 0, "xyz789", Role::Qa),
        "ticket_t-7_0_xyz789_qa"
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
    // Both provider_configured() disjuncts (key, keyless custom endpoint)
    // must claim a Pending workspace. Rows carry their own attribution.
    // (provider_key, custom_endpoint, ws_path, ws_name, claim_expect, status_msg)
    let cases = [
        (
            Some("sk-test"),
            None,
            "/tmp/test_pickup_claim",
            "ws_pickup_claim",
            "claim should succeed",
            "provider key configured → pending workspace claimed into discovery",
        ),
        // A keyless custom endpoint counts as provider configured — the
        // runtime honors a persisted custom endpoint, so without an OpenRouter
        // key the pickup must still claim the workspace into discovery.
        (
            None,
            Some("http://localhost:8080/v1"),
            "/tmp/test_pickup_endpoint",
            "ws_pickup_endpoint",
            "a persisted custom endpoint without a key must count as provider configured",
            "keyless custom endpoint → pending workspace claimed into discovery",
        ),
    ];
    for (provider_key, custom_endpoint, ws_path, ws_name, claim_expect, status_msg) in cases {
        // Per-row ConfigGuard: each row must run against its own disjunct.
        let _cfg = ConfigGuard::new(provider_key, custom_endpoint);
        let ws = create_test_workspace(ws_path, ws_name).await;

        let claimed = pickup_claim(&ws).await;
        let (generation, discover_diagnostics) = claimed.expect(claim_expect);
        assert_eq!(generation, 0, "fresh workspace has discovery_generation 0");
        assert!(
            discover_diagnostics,
            "no diagnostics exist yet → first discovery must run diagnostics"
        );

        let stored = crate::workspace::store()
            .get_by_name(ws_name)
            .await
            .expect("fetch")
            .expect("exists");
        assert_eq!(stored.status, WorkspaceStatus::Analyzing, "{}", status_msg);
        assert!(
            stored.paused,
            "the claim must set the analysis pause (blocks pipeline claims while discovery runs)"
        );
    }
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
