use super::*;
use crate::Role;
use crate::Tool;
use crate::Workspace;
use crate::role::DIAGNOSTICS_ROLE;
use crate::util::test::TicketBuilder;
use crate::util::test::assert_superseded_ticket;
use crate::util::test::expect_ticket;
use crate::util::test::init_test_stores;
use crate::util::test::make_ticket;
use crate::workspace::test_ws;
use crate::workspace::test_ws_named;
use strum::IntoEnumIterator;
use tempfile::TempDir;

/// Scenarios for testing invalid prerequisite/supersede inputs.
enum InvalidInputScenario {
    /// Prerequisite/supersede references a nonexistent ticket.
    NonExistent,
    /// Prerequisite/supersede references a ticket in a different workspace.
    CrossWorkspace,
    /// Prerequisite/supersede references the ticket itself (self-reference).
    SelfReference,
}

struct Case {
    name: &'static str,
    scenario: InvalidInputScenario,
}

/// Open a test store and create a default ticket.
/// Returns (store, temp_dir, ticket_id).
async fn setup() -> (BoardStore, TempDir, String) {
    let (store, tmp) = open_test_store().await;
    let id = make_ticket(
        &store,
        &test_ws_named("/ws", "ws"),
        "Test",
        TicketPhase::Backlog,
    )
    .await;
    (store, tmp, id)
}

#[tokio::test]
async fn test_get_ticket_phase() {
    let (store, _tmp) = open_test_store().await;

    // Non-existent ticket returns None.
    assert!(
        store
            .get_ticket_phase("nonexistent")
            .await
            .expect("query")
            .is_none()
    );

    let id = make_ticket(
        &store,
        &crate::workspace::test_ws_named("/workspace", "workspace"),
        "Status Test",
        TicketPhase::Planning,
    )
    .await;

    let phase = crate::util::test::expect_ticket_phase(&store, &id).await;
    assert_eq!(phase, TicketPhase::Planning);

    // After transition, reflects new status.
    store
        .transition_to(&id, None, TicketPhase::ReadyForDevelopment, None)
        .await
        .expect("set");
    let phase = crate::util::test::expect_ticket_phase(&store, &id).await;
    assert_eq!(phase, TicketPhase::ReadyForDevelopment);
}

#[test]
fn test_ticket_phase_parse_and_roundtrip() {
    // Roundtrip: as_ref() -> parse() for every variant
    for v in TicketPhase::iter() {
        let parsed: TicketPhase = v.as_ref().parse().unwrap();
        assert_eq!(&parsed, &v, "roundtrip failed for {v}");
    }

    // Error case
    assert!("unknown_phase".parse::<TicketPhase>().is_err());
}

#[test]
fn test_display_name_no_underscores() {
    // Every variant's display_name() must be underscore-free
    // and non-empty.
    for variant in TicketPhase::iter() {
        let name = variant.display_name();
        assert!(!name.is_empty(), "empty display_name for {variant}");
        assert!(
            !name.contains('_'),
            "display_name for {variant} still has underscore: {name}"
        );
    }
}

#[test]
fn test_ticket_phase_from_str_error_message() {
    let err = "bogus_status".parse::<TicketPhase>().unwrap_err();
    let msg = format!("{err}");

    assert!(
        msg.contains("Invalid phase"),
        "error should mention 'Invalid phase', got: {msg}"
    );
    assert!(
        msg.contains("bogus_status"),
        "error should contain the invalid input value, got: {msg}"
    );
    assert!(
        msg.contains("backlog"),
        "error should list valid phases (e.g. backlog), got: {msg}"
    );
    assert!(
        msg.contains("cancelled"),
        "error should list valid phases (e.g. cancelled), got: {msg}"
    );
}

#[tokio::test]
async fn test_unconditional_transition_clears_assignment() {
    let (store, _tmp, id) = setup().await;

    // Claim the ticket (sets assigned_to to NULL by default)
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::InDevelopment,
            "ws",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim")
        .expect("ticket exists");

    // Set assigned_to explicitly (matching production dispatch_engineer behavior)
    store
        .set_assigned_to(&claimed.id, Some(Role::Engineer.as_str()))
        .await
        .expect("set_assigned_to");
    let ticket = store
        .get_ticket(&id)
        .await
        .expect("get")
        .expect("should exist");
    assert!(
        ticket.assigned_to.is_some(),
        "assigned_to should be set after set_assigned_to"
    );

    // Update status — this should clear assigned_to
    store
        .transition_to(&id, None, TicketPhase::DiagnosticsDone, None)
        .await
        .expect("update");

    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::DiagnosticsDone);
    assert!(
        ticket.assigned_to.is_none(),
        "assigned_to should be cleared after unconditional transition"
    );
}

#[tokio::test]
async fn test_guarded_transition() {
    let (store, _tmp, id) = setup().await;

    // Wrong expected phase — should fail, ticket unchanged.
    let result = store
        .transition_to(
            &id,
            Some(TicketPhase::Done),
            TicketPhase::InDevelopment,
            None,
        )
        .await;
    assert!(
        result.is_err(),
        "guarded transition with wrong phase should fail"
    );
    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::Backlog);

    // Correct expected phase — should succeed.
    store
        .transition_to(
            &id,
            Some(TicketPhase::Backlog),
            TicketPhase::InDevelopment,
            None,
        )
        .await
        .expect("guarded transition with correct phase should succeed");
    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::InDevelopment);
}

#[tokio::test]
async fn test_add_comment() {
    let (store, _tmp, id) = setup().await;

    store
        .add_comment(&id, Role::Engineer.as_str(), "done!")
        .await
        .expect("add comment");

    let comments = store.get_comments(&id).await.expect("get comments");
    assert_eq!(comments.len(), 1);
    assert_eq!(comments[0].role, Role::Engineer.as_str());
    assert_eq!(comments[0].content, "done!");
    assert!(!comments[0].created_at.is_empty());

    // Verify updated_at was bumped
    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert!(ticket.updated_at > ticket.created_at);
}

#[tokio::test]
async fn test_list_tickets() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    make_ticket(&store, &ws, "A", TicketPhase::Backlog).await;
    make_ticket(&store, &ws, "B", TicketPhase::Backlog).await;
    make_ticket(&store, &ws, "C", TicketPhase::Backlog).await;

    // All tickets for the workspace
    let tickets = store
        .list_all_tickets(Some("ws"), None)
        .await
        .expect("list");
    assert_eq!(tickets.len(), 3);

    // Filter by status (none match since all are Backlog)
    let tickets = store
        .list_all_tickets(Some("ws"), Some(TicketPhase::Done))
        .await
        .expect("list");
    assert_eq!(tickets.len(), 0);
}

/// Verify that `reset_inflight_tickets` correctly transitions each in-flight
/// ticket phase back to its ready state, and that non-inflight phases (e.g.
/// Backlog) are left untouched.
#[tokio::test]
async fn test_reset_inflight_tickets() {
    /// A single reset transition case.
    struct Case {
        name: &'static str,
        /// Unique suffix for workspace names (isolates cases).
        suffix: &'static str,
        /// The phase the ticket starts in.
        start: TicketPhase,
        /// The expected phase after reset.
        expected: TicketPhase,
        /// Expected pipeline_reservation after reset.
        reservation: bool,
    }

    let cases = [
        Case {
            name: "Backlog unaffected (not an inflight phase)",
            suffix: "a",
            start: TicketPhase::Backlog,
            expected: TicketPhase::Backlog,
            reservation: false,
        },
        Case {
            name: "Analysis → Backlog (no reservation)",
            suffix: "b",
            start: TicketPhase::Analysis,
            expected: TicketPhase::Backlog,
            reservation: false,
        },
        Case {
            name: "InDevelopment → ReadyForDevelopment (reservation=1)",
            suffix: "c",
            start: TicketPhase::InDevelopment,
            expected: TicketPhase::ReadyForDevelopment,
            reservation: true,
        },
        Case {
            name: "InDiagnostics → ReadyForDevelopment (reservation=1)",
            suffix: "d",
            start: TicketPhase::InDiagnostics,
            expected: TicketPhase::ReadyForDevelopment,
            reservation: true,
        },
        Case {
            name: "InSanitation → QaPassed (reservation=1)",
            suffix: "e",
            start: TicketPhase::InSanitation,
            expected: TicketPhase::QaPassed,
            reservation: true,
        },
        Case {
            name: "InQa → Reviewed (no reservation)",
            suffix: "f",
            start: TicketPhase::InQa,
            expected: TicketPhase::Reviewed,
            reservation: false,
        },
        Case {
            name: "InReview → DiagnosticsDone (no reservation)",
            suffix: "g",
            start: TicketPhase::InReview,
            expected: TicketPhase::DiagnosticsDone,
            reservation: false,
        },
    ];

    let (store, _tmp) = open_test_store().await;

    for case in &cases {
        let ws = test_ws_named(&format!("/{}", case.suffix), case.suffix);

        let id = make_ticket(&store, &ws, case.name, case.start).await;

        store.reset_inflight_tickets().await.expect("reset");

        let t = expect_ticket(&store, &id).await;
        assert_eq!(
            t.phase, case.expected,
            "Case '{}': unexpected status after reset",
            case.name,
        );
        assert_eq!(
            t.pipeline_reservation, case.reservation,
            "Case '{}': unexpected pipeline_reservation after reset",
            case.name,
        );
        assert!(
            t.assigned_to.is_none(),
            "Case '{}': assigned_to should be NULL after reset",
            case.name,
        );
    }
}

#[tokio::test]
async fn test_claim_prefers_reserved_ticket() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // Create two ReadyForDevelopment tickets
    let fresh_id = make_ticket(&store, &ws, "Fresh", TicketPhase::ReadyForDevelopment).await;
    let reserved_id = make_ticket(&store, &ws, "Reserved", TicketPhase::ReadyForDevelopment).await;

    // Set reservation on the second ticket
    store
        .transition_to(
            &reserved_id,
            Some(TicketPhase::ReadyForDevelopment),
            TicketPhase::ReadyForDevelopment,
            Some(true),
        )
        .await
        .expect("set reservation");

    // When claiming with PipelineCheck::Enforce, the reserved ticket should be picked first
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::ReadyForDevelopment,
            TicketPhase::InDevelopment,
            "ws",
            PipelineCheck::Enforce,
        )
        .await
        .expect("claim")
        .expect("should claim a ticket");
    assert_eq!(
        claimed.id, reserved_id,
        "Reserved ticket should be claimed before fresh one"
    );
    assert!(
        !claimed.pipeline_reservation,
        "Claim should clear pipeline_reservation"
    );

    // Verify the cleared reservation is persisted in the DB
    // (the returned Ticket struct already reflects the DB state, but
    // a separate re-read explicitly tests persistence).
    let reserved_db = expect_ticket(&store, &reserved_id).await;
    assert!(
        !reserved_db.pipeline_reservation,
        "Reservation should be 0 in DB after claim"
    );

    // After the reserved ticket is claimed (now InDevelopment, pipeline-blocking),
    // the fresh ticket is still at ReadyForDevelopment but cannot be claimed
    // because the pipeline is blocked. Verify the fresh ticket remains untouched.
    let fresh = expect_ticket(&store, &fresh_id).await;
    assert_eq!(
        fresh.phase,
        TicketPhase::ReadyForDevelopment,
        "Fresh ticket should still be at ReadyForDevelopment"
    );
    assert!(
        !fresh.pipeline_reservation,
        "Fresh ticket should have no reservation"
    );
}

#[tokio::test]
async fn test_has_pipeline_blocker_reserved() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // A fresh ReadyForDevelopment ticket should NOT be a blocker
    let id = make_ticket(&store, &ws, "Fresh", TicketPhase::ReadyForDevelopment).await;
    assert!(
        !store
            .has_pipeline_blocker_for_workspace("ws")
            .await
            .expect("check"),
        "Fresh ReadyForDevelopment ticket should not be a pipeline blocker"
    );

    // After setting reservation, it should be a blocker
    store
        .transition_to(
            &id,
            Some(TicketPhase::ReadyForDevelopment),
            TicketPhase::ReadyForDevelopment,
            Some(true),
        )
        .await
        .expect("set reservation");
    assert!(
        store
            .has_pipeline_blocker_for_workspace("ws")
            .await
            .expect("check"),
        "Reserved ReadyForDevelopment ticket should be a pipeline blocker"
    );

    // After removing reservation, it should not be a blocker
    store
        .transition_to(
            &id,
            Some(TicketPhase::ReadyForDevelopment),
            TicketPhase::ReadyForDevelopment,
            Some(false),
        )
        .await
        .expect("clear reservation");
    assert!(
        !store
            .has_pipeline_blocker_for_workspace("ws")
            .await
            .expect("check"),
        "Non-reserved ReadyForDevelopment ticket should not be a pipeline blocker again"
    );
}

/// Assert that [`BoardStore::has_active_tickets_excluding`] returns the
/// expected value. Supports both static and formatted messages.
async fn assert_active_excluding(
    store: &BoardStore,
    ws_name: &str,
    exclude_id: &str,
    expected: bool,
    msg: impl std::fmt::Display,
) {
    assert_eq!(
        store
            .has_active_tickets_excluding(ws_name, exclude_id)
            .await
            .expect("check"),
        expected,
        "{msg}"
    );
}

/// Create 5 tickets in non-active phases under workspace "ws_non" (/ws_non),
/// returning their IDs.
///
/// Non-active phases covered: Done, Cancelled, Failed, Planning, Backlog.
/// Note: Analysis is also filtered out by the SQL query but is intentionally
/// omitted here — it has its own dedicated test coverage elsewhere.
async fn create_non_active_tickets(store: &BoardStore) -> Vec<String> {
    let ws = test_ws_named("/ws_non", "ws_non");
    vec![
        make_ticket(store, &ws, "Done", TicketPhase::Done).await,
        make_ticket(store, &ws, "Cancelled", TicketPhase::Cancelled).await,
        make_ticket(store, &ws, "Failed", TicketPhase::Failed).await,
        make_ticket(store, &ws, "Planning", TicketPhase::Planning).await,
        make_ticket(store, &ws, "Backlog", TicketPhase::Backlog).await,
    ]
}

/// Verify that [`BoardStore::has_active_tickets_excluding`] correctly identifies
/// active tickets (PIPELINE_BLOCKING_PHASES + ReadyForDevelopment) per workspace,
/// excluding a specified ticket ID.
///
/// Active tickets include all ReadyForDevelopment tickets regardless of
/// `pipeline_reservation`, unlike [`has_pipeline_blocker_for_workspace`] which
/// requires `pipeline_reservation = 1`. This is intentional — unstarted backlog
/// tickets are considered active to suppress Done notifications until the pipeline
/// is fully drained.
#[tokio::test]
async fn test_has_active_tickets_excluding() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // Create one ticket per active status: all PIPELINE_BLOCKING_PHASES + ReadyForDevelopment
    let rfd_id = make_ticket(&store, &ws, "RFD", TicketPhase::ReadyForDevelopment).await;
    let in_dev_id = make_ticket(&store, &ws, "InDev", TicketPhase::InDevelopment).await;
    let done_id = make_ticket(&store, &ws, "Done", TicketPhase::Done).await;
    let cancelled_id = make_ticket(&store, &ws, "Cancelled", TicketPhase::Cancelled).await;

    // All non-excluded active tickets are found
    assert_active_excluding(
        &store,
        "ws",
        &done_id,
        true,
        "Should find active tickets (RFD + InDev) when excluding Done",
    )
    .await;

    // Excluding an active ticket still finds another active ticket
    assert_active_excluding(
        &store,
        "ws",
        &rfd_id,
        true,
        "Should find InDev as active when excluding RFD",
    )
    .await;
    assert_active_excluding(
        &store,
        "ws",
        &in_dev_id,
        true,
        "Should find RFD as active when excluding InDev",
    )
    .await;

    // Non-active (Done, Cancelled) exclusion should still find active tickets
    for exclude in [&done_id, &cancelled_id] {
        assert_active_excluding(
            &store,
            "ws",
            exclude,
            true,
            "Non-active exclusion should still find active tickets",
        )
        .await;
    }

    // ReadyForDevelopment without reservation counts as active
    // (rfd_id already has no reservation — it was created with default)
    assert_active_excluding(
        &store,
        "ws",
        "nonexistent",
        true,
        "Should find active tickets for nonexistent exclude ID",
    )
    .await;

    // Different workspace — no tickets
    assert_active_excluding(
        &store,
        "other_ws",
        &rfd_id,
        false,
        "Should not find active tickets in unrelated workspace",
    )
    .await;

    // Workspace with only non-active tickets — Done, Cancelled, Failed, Planning, Backlog
    let non_active_ids = create_non_active_tickets(&store).await;
    for exclude in &non_active_ids {
        assert_active_excluding(
                &store,
                "ws_non",
                exclude,
                false,
                format!("Workspace with only non-active tickets should have no active tickets (excluded {exclude})"),
            )
            .await;
    }
    // Excluding a nonexistent ID in a non-active-only workspace also returns false
    assert_active_excluding(
        &store,
        "ws_non",
        "nonexistent",
        false,
        "No active tickets for nonexistent exclude ID in non-active-only workspace",
    )
    .await;
}

/// Verify that every non-transitory pipeline-blocking phase has a reset transition.
///
/// [`PIPELINE_BLOCKING_PHASES`] defines 9 phases; 5 of them (InDevelopment,
/// InDiagnostics, InSanitation, InReview, InQa) have entries in
/// [`RESET_TRANSITIONS`]. The remaining 4 phases
/// ([`TRANSITORY_HANDOFF_PHASES`]) are transitory handoff states that the
/// poller picks up within seconds — no agent is mid-execution in those states,
/// so they don't need reset entries.
///
/// This test does NOT assert the reverse direction (reset → pipeline blocker),
/// because [`RESET_TRANSITIONS`] also includes `Analysis → Backlog`, and `Analysis`
/// is intentionally not a pipeline blocker (it's a pre-flight phase).
///
/// It also mechanically verifies that [`TRANSITORY_HANDOFF_PHASES`] is a subset of
/// [`PIPELINE_BLOCKING_PHASES`], ensuring the two sets stay in sync.
#[test]
fn test_pipeline_blockers_coverage() {
    // Verify that every transitory handoff phase is a pipeline blocker.
    for phase in TRANSITORY_HANDOFF_PHASES {
        assert!(
            PIPELINE_BLOCKING_PHASES.contains(phase),
            "\
TRANSITORY_HANDOFF_PHASES contains `{phase}` which is not in \
PIPELINE_BLOCKING_PHASES. Every transitory handoff phase must also \
be a pipeline blocker.\
                ",
        );
    }

    // Collect all `from` phases from BoardStore::RESET_TRANSITIONS for easy lookup.
    let reset_from: Vec<TicketPhase> = BoardStore::RESET_TRANSITIONS
        .iter()
        .map(|t| t.from)
        .collect();

    for phase in PIPELINE_BLOCKING_PHASES {
        let has_reset = reset_from.contains(phase);
        assert!(
            has_reset || phase.is_transitory_handoff(),
            "\
PIPELINE_BLOCKING_PHASES contains `{phase}` which has no corresponding \
entry in RESET_TRANSITIONS and is not a transitory handoff phase \
(see `TicketPhase::is_transitory_handoff`). Either add a reset transition to \
RESET_TRANSITIONS, or mark the phase as transitory handoff in that method \
with a comment explaining why no agent is mid-execution in that state.\
                ",
        );
    }
}

#[tokio::test]
async fn test_claim_ticket_in_workspace() {
    let (store, _tmp) = open_test_store().await;

    // Create tickets in two different workspaces
    let ws_a = test_ws_named("/ws_a", "workspace_a");
    let ws_b = test_ws_named("/ws_b", "workspace_b");

    let id_a = make_ticket(&store, &ws_a, "Ticket A", TicketPhase::Backlog).await;

    let id_b = make_ticket(&store, &ws_b, "Ticket B", TicketPhase::Backlog).await;

    // Claim ticket from workspace A — should succeed
    let claimed_a = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::InDevelopment,
            "workspace_a",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim in ws_a")
        .expect("should claim ticket from ws_a");
    assert_eq!(claimed_a.id, id_a);
    assert_eq!(claimed_a.workspace_name, "workspace_a");
    assert_eq!(claimed_a.phase, TicketPhase::InDevelopment);
    assert!(claimed_a.assigned_to.is_none());

    // Claim from workspace A again — should return None (no more backlog tickets)
    assert!(
        store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::InDevelopment,
                "workspace_a",
                PipelineCheck::Skip,
            )
            .await
            .expect("second claim in ws_a")
            .is_none(),
        "no more tickets to claim in ws_a"
    );

    // Claim ticket from workspace B — should still succeed (different workspace)
    let claimed_b = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::InDevelopment,
            "workspace_b",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim in ws_b")
        .expect("should claim ticket from ws_b");
    assert_eq!(claimed_b.id, id_b);
    assert_eq!(claimed_b.workspace_name, "workspace_b");
}

/// Table-driven tests for [`PipelineCheck::Enforce`] — claims with pipeline occupancy
/// checking enabled.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn test_claim_ticket_in_workspace_if_pipeline_free() {
    /// The pipeline scenario for a single test case.
    enum Scenario {
        /// Blocker in the same workspace — claim should be blocked.
        SameWorkspace(TicketPhase),
        /// Blocker in a different workspace — claim should succeed.
        DifferentWorkspace(TicketPhase),
        /// No blocker — claim should succeed.
        NoBlocker,
    }

    struct Case {
        name: &'static str,
        /// Unique suffix for workspace names (isolates cases).
        suffix: &'static str,
        scenario: Scenario,
    }

    let cases = [
        Case {
            name: "blocked by same-workspace pipeline ticket",
            suffix: "blocked",
            scenario: Scenario::SameWorkspace(TicketPhase::InReview),
        },
        Case {
            name: "not blocked by cross-workspace pipeline ticket",
            suffix: "cross",
            scenario: Scenario::DifferentWorkspace(TicketPhase::InDevelopment),
        },
        Case {
            name: "no blocker succeeds",
            suffix: "none",
            scenario: Scenario::NoBlocker,
        },
    ];

    let (store, _tmp) = open_test_store().await;

    for case in &cases {
        let suffix = case.suffix;

        // Derive workspace names from the scenario.
        let (claim_ws_name, blocker_ws_name) = match &case.scenario {
            Scenario::DifferentWorkspace(_) => (
                format!("ws_{suffix}_claimable"),
                format!("ws_{suffix}_blocker"),
            ),
            // SameWorkspace and NoBlocker both use a single workspace name.
            Scenario::SameWorkspace(_) | Scenario::NoBlocker => {
                let name = format!("ws_{suffix}");
                (name.clone(), name)
            }
        };

        let expected_claim = !matches!(case.scenario, Scenario::SameWorkspace(_));

        let blocker_ws = test_ws_named(&format!("/{blocker_ws_name}"), &blocker_ws_name);
        let claimable_ws = test_ws_named(&format!("/{claim_ws_name}"), &claim_ws_name);

        // Create a pipeline blocker (if any)
        if let Scenario::SameWorkspace(phase) | Scenario::DifferentWorkspace(phase) = &case.scenario
        {
            // When blocker and claimable share a workspace, place the
            // blocker in the claimable's workspace (they are the same).
            let blocker_target = match &case.scenario {
                Scenario::DifferentWorkspace(_) => &blocker_ws,
                Scenario::SameWorkspace(_) => &claimable_ws,
                // Not reachable: NoBlocker is guarded by the enclosing if-let.
                Scenario::NoBlocker => unreachable!(),
            };
            make_ticket(&store, blocker_target, "Blocker", *phase).await;
        }

        // Create a claimable ticket
        let id = make_ticket(
            &store,
            &claimable_ws,
            "Claimable",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Claim with PipelineCheck::Enforce
        let claimed = store
            .claim_ticket_in_workspace(
                TicketPhase::ReadyForDevelopment,
                TicketPhase::InDevelopment,
                &claim_ws_name,
                PipelineCheck::Enforce,
            )
            .await
            .expect("claim should not error");

        if expected_claim {
            let claimed = claimed.expect("should claim ticket");
            assert_eq!(claimed.id, id, "Case '{}': wrong ticket id", case.name);
            assert_eq!(
                claimed.phase,
                TicketPhase::InDevelopment,
                "Case '{}': wrong status after claim",
                case.name
            );
        } else {
            assert!(
                claimed.is_none(),
                "Case '{}': claim should be blocked",
                case.name
            );
        }
    }
}

// ── Prerequisites ────────────────────────────────────────────

#[tokio::test]
async fn test_create_ticket_with_prerequisites() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // Create prerequisite tickets first
    let p1 = make_ticket(&store, &ws, "P1", TicketPhase::Backlog).await;
    let p2 = make_ticket(&store, &ws, "P2", TicketPhase::Backlog).await;

    // Create a ticket depending on both
    let deps = vec![p1.clone(), p2.clone()];
    let id = TicketBuilder::new(&store, &ws)
        .title("Dependent")
        .desc("needs both")
        .prereqs(&deps)
        .create()
        .await
        .expect("create dependent");

    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert_eq!(ticket.prerequisites.len(), 2);
    assert!(ticket.prerequisites.contains(&p1));
    assert!(ticket.prerequisites.contains(&p2));
}

/// Table-driven tests for `create_ticket` with invalid prerequisite inputs.
#[tokio::test]
async fn test_create_ticket_invalid_inputs() {
    let cases = [
        Case {
            name: "nonexistent prerequisite",
            scenario: InvalidInputScenario::NonExistent,
        },
        Case {
            name: "self-referencing prerequisite",
            scenario: InvalidInputScenario::SelfReference,
        },
        Case {
            name: "cross-workspace prerequisite",
            scenario: InvalidInputScenario::CrossWorkspace,
        },
    ];

    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");
    let ws_b = test_ws_named("/ws_b", "ws_b");
    // Isolated workspace for SelfReference — its own counter avoids
    // ordering dependencies with CrossWorkspace which also creates
    // seeds in `ws`.
    let ws_sr = test_ws_named("/ws_sr", "ws_sr");

    for case in &cases {
        let expected_error = match case.scenario {
            InvalidInputScenario::NonExistent => "not found",
            InvalidInputScenario::SelfReference => "cannot depend on itself",
            InvalidInputScenario::CrossWorkspace => "Cross-workspace",
        };

        // Create a seed ticket for scenarios that need one.
        // NonExistent: no seed needed — uses a nonexistent ID directly.
        // CrossWorkspace: create a ticket in `ws` to use as a
        //   cross-workspace prerequisite for a ticket in `ws_b`.
        // SelfReference: create exactly one ticket in its own workspace
        //   `ws_sr` to advance the counter so the next ticket will
        //   have ID `ws_sr-1`.
        let seed: Option<String> = match &case.scenario {
            InvalidInputScenario::NonExistent => None,
            InvalidInputScenario::CrossWorkspace => {
                Some(make_ticket(&store, &ws, "Existing", TicketPhase::Backlog).await)
            }
            InvalidInputScenario::SelfReference => {
                Some(make_ticket(&store, &ws_sr, "First", TicketPhase::Backlog).await)
            }
        };

        let target_ws = match case.scenario {
            InvalidInputScenario::CrossWorkspace => &ws_b,
            InvalidInputScenario::SelfReference => &ws_sr,
            InvalidInputScenario::NonExistent => &ws,
        };

        // Build prerequisites for each scenario.
        // NonExistent: a nonexistent ticket ID.
        // CrossWorkspace: the ticket created in `ws` (different workspace).
        // SelfReference: hardcoded `{ws_sr}-1` — the ID the next ticket
        //   receives in the isolated workspace, creating a self-reference.
        let prereqs: Vec<String> = match &case.scenario {
            InvalidInputScenario::NonExistent => vec!["nonexistent-1".to_string()],
            InvalidInputScenario::CrossWorkspace => {
                vec![seed.clone().expect("seed must exist for CrossWorkspace")]
            }
            InvalidInputScenario::SelfReference => {
                // After creating exactly one seed ticket above, the next
                // ticket in this isolated workspace receives ID `ws_sr-1`.
                vec![format!("{}-1", ws_sr.name)]
            }
        };

        let err = TicketBuilder::new(&store, target_ws)
            .title("New")
            .prereqs(&prereqs)
            .create()
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains(expected_error),
            "Case '{}': expected error containing '{}', got: {err}",
            case.name,
            expected_error
        );
    }
}

/// Create a 2-ticket dependency chain: A (no prereqs) → B (depends on A).
async fn create_chain_ab(store: &BoardStore, ws: Workspace) -> (String, String) {
    let a = make_ticket(store, &ws, "A", TicketPhase::Backlog).await;
    let b = TicketBuilder::new(store, &ws)
        .title("B")
        .desc("depends on A")
        .prereqs(std::slice::from_ref(&a))
        .create()
        .await
        .expect("create b");
    (a, b)
}

#[tokio::test]
async fn test_circular_dependency_rejected() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    let (a, b) = create_chain_ab(&store, ws.clone()).await;

    // Verify that A→B chain works: creating a ticket with both A and B
    // as prerequisites is NOT a cycle (it's just redundant, since A is
    // already transitively required through B). This should succeed.
    let _c = TicketBuilder::new(&store, &ws)
        .title("C")
        .desc("depends on both")
        .prereqs(&[a.clone(), b.clone()])
        .create()
        .await
        .expect("create c — A and B as prereqs is not a cycle");
}

#[tokio::test]
async fn test_transitive_prerequisites_block() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    let (a, b) = create_chain_ab(&store, ws.clone()).await;

    // C depends on B
    let c = TicketBuilder::new(&store, &ws)
        .title("C")
        .desc("top")
        .prereqs(std::slice::from_ref(&b))
        .create()
        .await
        .expect("create c");

    // C should be blocked even though B is done — A is still blocking
    // First claim: A is the only unblocked one
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim")
        .expect("should claim A");
    assert_eq!(claimed.id, a);

    // B should still be blocked — A is in Analysis, not Done yet
    let second = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim");
    assert!(
        second.is_none(),
        "B should be blocked because A is in Analysis, not Done"
    );

    // Move A to done
    store
        .transition_to(&a, None, TicketPhase::Done, None)
        .await
        .expect("done a");

    // Now B should be claimable
    let claimed2 = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim")
        .expect("should claim B");
    assert_eq!(claimed2.id, b);

    // Move B to done
    store
        .transition_to(&b, None, TicketPhase::Done, None)
        .await
        .expect("done b");

    // Now C should be claimable
    let claimed3 = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
        )
        .await
        .expect("claim")
        .expect("should claim C");
    assert_eq!(claimed3.id, c);
}

async fn assert_archive_empty_db(store: &BoardStore) {
    let count = store
        .archive_stale_cancelled(1)
        .await
        .expect("archive_stale_cancelled");
    assert_eq!(count, 0, "Empty DB stale archive should return 0");
    let count = store
        .archive_all_done_and_cancelled(None)
        .await
        .expect("archive_all_done_and_cancelled");
    assert_eq!(count, 0, "Empty DB all archive should return 0");
}

#[tokio::test]
async fn test_archive_stale_cancelled() {
    let (store, _tmp) = open_test_store().await;
    assert_archive_empty_db(&store).await;

    let ws = test_ws_named("/ws", "ws");

    // Ticket 1: cancelled, old (2h) → should be archived
    let two_hours_ago = (Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
    let old_cancelled_id = make_ticket(&store, &ws, "old-cancelled", TicketPhase::Cancelled).await;
    store
        .conn
        .execute(
            "UPDATE tickets SET updated_at = ?1 WHERE id = ?2",
            crate::turso::params![two_hours_ago.clone(), old_cancelled_id.clone()],
        )
        .await
        .expect("backdate");

    // Ticket 2: cancelled, fresh → should NOT be archived
    let fresh_cancelled_id =
        make_ticket(&store, &ws, "fresh-cancelled", TicketPhase::Cancelled).await;
    // No backdating — updated_at is now.

    // Ticket 3: not cancelled (Backlog), old → should NOT be archived
    let old_backlog_id = make_ticket(&store, &ws, "old-backlog", TicketPhase::Backlog).await;
    store
        .conn
        .execute(
            "UPDATE tickets SET updated_at = ?1 WHERE id = ?2",
            crate::turso::params![two_hours_ago.clone(), old_backlog_id.clone()],
        )
        .await
        .expect("backdate");

    // Act
    let count = store
        .archive_stale_cancelled(1)
        .await
        .expect("archive_stale_cancelled");
    assert_eq!(count, 1, "should archive only the old cancelled ticket");

    // Assert
    let old_cancelled = crate::util::test::expect_ticket(&store, &old_cancelled_id).await;
    assert!(
        old_cancelled.is_archived,
        "old cancelled ticket should be archived"
    );
    assert_eq!(old_cancelled.phase, TicketPhase::Cancelled);

    let fresh_cancelled = crate::util::test::expect_ticket(&store, &fresh_cancelled_id).await;
    assert!(
        !fresh_cancelled.is_archived,
        "fresh cancelled ticket should NOT be archived"
    );
    assert_eq!(fresh_cancelled.phase, TicketPhase::Cancelled);

    let old_backlog = crate::util::test::expect_ticket(&store, &old_backlog_id).await;
    assert!(
        !old_backlog.is_archived,
        "old non-cancelled ticket should NOT be archived"
    );
    assert_eq!(old_backlog.phase, TicketPhase::Backlog);
}

#[tokio::test]
async fn test_archive_all_done_and_cancelled() {
    let (store, _tmp) = open_test_store().await;
    assert_archive_empty_db(&store).await;

    let ws = test_ws_named("/ws", "ws");

    // Create three tickets: one Done, one Cancelled, one Backlog.
    let done_id = make_ticket(&store, &ws, "done", TicketPhase::Done).await;

    let cancelled_id = make_ticket(&store, &ws, "cancelled", TicketPhase::Cancelled).await;

    let backlog_id = make_ticket(&store, &ws, "backlog", TicketPhase::Backlog).await;
    // Leave in Backlog.

    // Act
    let count = store
        .archive_all_done_and_cancelled(None)
        .await
        .expect("archive");
    assert_eq!(count, 2, "should archive Done and Cancelled tickets");

    // Assert
    let done_ticket = crate::util::test::expect_ticket(&store, &done_id).await;
    assert!(done_ticket.is_archived, "Done ticket should be archived");
    assert_eq!(done_ticket.phase, TicketPhase::Done);

    let cancelled_ticket = crate::util::test::expect_ticket(&store, &cancelled_id).await;
    assert!(
        cancelled_ticket.is_archived,
        "Cancelled ticket should be archived"
    );
    assert_eq!(cancelled_ticket.phase, TicketPhase::Cancelled);

    let backlog_ticket = crate::util::test::expect_ticket(&store, &backlog_id).await;
    assert!(
        !backlog_ticket.is_archived,
        "Backlog ticket should NOT be archived"
    );
    assert_eq!(backlog_ticket.phase, TicketPhase::Backlog);
}

#[tokio::test]
async fn test_archive_all_done_and_cancelled_workspace_filter() {
    let (store, _tmp) = open_test_store().await;

    // Create a done ticket in ws1 and another in ws2.
    let id1 = make_ticket(
        &store,
        &test_ws_named("/ws1", "ws1"),
        "Test",
        TicketPhase::Done,
    )
    .await;
    let id2 = make_ticket(
        &store,
        &test_ws_named("/ws2", "ws2"),
        "Test",
        TicketPhase::Done,
    )
    .await;

    // Archive only ws1.
    let count = store
        .archive_all_done_and_cancelled(Some("ws1"))
        .await
        .expect("archive_all_done_and_cancelled");
    assert_eq!(count, 1, "Should archive only ws1 ticket");

    let ticket1 = crate::util::test::expect_ticket(&store, &id1).await;
    assert!(ticket1.is_archived, "ws1 ticket should be archived");
    assert_eq!(
        ticket1.phase,
        TicketPhase::Done,
        "ws1 phase should remain Done"
    );

    let ticket2 = crate::util::test::expect_ticket(&store, &id2).await;
    assert!(!ticket2.is_archived, "ws2 ticket should NOT be archived");
    assert_eq!(
        ticket2.phase,
        TicketPhase::Done,
        "ws2 ticket should remain Done"
    );
}

#[tokio::test]
async fn test_count_by_phase_excludes_archived() {
    let (store, _tmp) = open_test_store().await;
    // Create a ticket set to Done.
    let _id = make_ticket(
        &store,
        &test_ws_named("/ws", "ws"),
        "Test",
        TicketPhase::Done,
    )
    .await;

    // Before archiving, count includes the Done ticket.
    let count_before = store
        .count_by_phase(TicketPhase::Done, None)
        .await
        .expect("count before");
    assert_eq!(count_before, 1, "Should count Done ticket before archive");

    // Archive done tickets.
    let archived = store
        .archive_all_done_and_cancelled(None)
        .await
        .expect("archive");
    assert_eq!(archived, 1, "Should have archived 1 ticket");

    // After archiving, count_by_phase(Done) should return 0.
    let count_after = store
        .count_by_phase(TicketPhase::Done, None)
        .await
        .expect("count after");
    assert_eq!(count_after, 0, "Should not count archived Done tickets");

    // Archived tickets with other statuses should also be excluded.
    let count_cancelled = store
        .count_by_phase(TicketPhase::Cancelled, None)
        .await
        .expect("count cancelled");
    assert_eq!(count_cancelled, 0, "No Cancelled tickets exist");
}

#[tokio::test]
async fn test_create_ticket_tool_with_prerequisites() {
    crate::util::test::init_test_stores().await;

    let store = crate::board::BOARD.get().unwrap();
    let ws = test_ws("/tmp/test_ws_tool_prereqs");

    // Create a prerequisite via the store directly
    let p_id = make_ticket(store, &ws, "Pre", TicketPhase::Backlog).await;

    let tool = crate::tools::CreateTicketTool::new("test");
    let args = serde_json::json!({
        "title": "Test with prereqs",
        "description": "depends on something",
        "prerequisites": [p_id],
    });
    let result = tool.execute(&ws, args).await.expect("execute");
    assert!(
        result.contains(&p_id),
        "Output should mention prerequisite ID"
    );
}

/// Supersede a live ticket (`Backlog` → `Cancelled`).
///
/// This also implicitly covers superseding an already-cancelled ticket: the
/// cancellation UPDATE (`supersede_and_create` line 797) has no phase guard
/// (`WHERE id = ?3` without `AND status = ?`), so it runs identically
/// regardless of the old ticket's current phase. A separate test with a
/// `Cancelled` starting phase would exercise the exact same SQL path and
/// assert the same invariants (`assert_superseded_ticket`, `supersedes`
/// back-link), making it redundant with this one.
#[tokio::test]
async fn test_supersede_and_create_basic() {
    init_test_stores().await;
    let store = crate::board::BOARD.get().unwrap();
    let ws = test_ws_named("/ws", "ws");
    let old_id = make_ticket(store, &ws, "Test", TicketPhase::Backlog).await;

    // Supersede it
    let new_id = TicketBuilder::new(store, &ws)
        .title("New title")
        .desc("New desc")
        .supersede(&old_id)
        .await
        .expect("supersede");

    // Old ticket is cancelled and points forward to the new ticket
    let old = expect_ticket(store, &old_id).await;
    assert_superseded_ticket(&old);
    assert_eq!(
        old.superseded_by.as_deref(),
        Some(new_id.as_str()),
        "superseded ticket should point to the new ticket"
    );

    // New ticket is in Backlog and links to old
    let new = expect_ticket(store, &new_id).await;
    assert_eq!(new.phase, TicketPhase::Backlog);
    assert_eq!(new.supersedes.as_deref(), Some(old_id.as_str()));
    assert_eq!(new.title, "New title");
}

#[tokio::test]
async fn test_supersede_rewires_only_matching_prerequisite() {
    init_test_stores().await;
    let store = crate::board::BOARD.get().unwrap();
    let ws = test_ws_named("/ws", "ws");

    // Create ticket A (will be superseded) and ticket C (independent).
    let a_id = make_ticket(store, &ws, "A", TicketPhase::Backlog).await;
    let c_id = make_ticket(store, &ws, "C", TicketPhase::Backlog).await;

    // Create ticket B that depends on both A and C.
    let b_id = TicketBuilder::new(store, &ws)
        .title("B")
        .desc("dep on A and C")
        .prereqs(&[a_id.clone(), c_id.clone()])
        .create()
        .await
        .expect("create B");

    // Create ticket D with no prerequisites — should be untouched.
    let d_id = make_ticket(store, &ws, "D", TicketPhase::Backlog).await;

    // Supersede A → A2.
    let supersede_id = TicketBuilder::new(store, &ws)
        .title("A2")
        .desc("refined")
        .supersede(&a_id)
        .await
        .expect("supersede");

    // B's prerequisites: A→A2, C unchanged.
    let b = store
        .get_ticket(&b_id)
        .await
        .expect("get B")
        .expect("B exists");
    assert_eq!(b.prerequisites, vec![supersede_id.clone(), c_id.clone()]);

    // D untouched.
    let d = store
        .get_ticket(&d_id)
        .await
        .expect("get D")
        .expect("D exists");
    assert!(d.prerequisites.is_empty());
}

/// Table-driven tests for `supersede_and_create` with invalid inputs.
#[tokio::test]
async fn test_supersede_invalid_inputs() {
    let cases = [
        Case {
            name: "nonexistent original",
            scenario: InvalidInputScenario::NonExistent,
        },
        Case {
            name: "cross-workspace supersede",
            scenario: InvalidInputScenario::CrossWorkspace,
        },
        Case {
            name: "self-referencing prerequisites",
            scenario: InvalidInputScenario::SelfReference,
        },
    ];

    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");
    let ws_b = test_ws_named("/ws_b", "ws_b");

    for case in &cases {
        let expected_error = match case.scenario {
            InvalidInputScenario::NonExistent => "not found",
            InvalidInputScenario::CrossWorkspace => "Cross-workspace",
            InvalidInputScenario::SelfReference => "supersede and depend",
        };

        let original_id = match case.scenario {
            InvalidInputScenario::NonExistent => None,
            InvalidInputScenario::CrossWorkspace | InvalidInputScenario::SelfReference => {
                let id = make_ticket(&store, &ws, "A", TicketPhase::Backlog).await;
                Some(id)
            }
        };

        let target_ws = match case.scenario {
            InvalidInputScenario::CrossWorkspace => &ws_b,
            InvalidInputScenario::NonExistent | InvalidInputScenario::SelfReference => &ws,
        };
        let supersede_id: &str = original_id.as_deref().unwrap_or("nonexistent");
        // prereqs include the original id only for SelfReference.
        let prereqs: Vec<String> = match &case.scenario {
            InvalidInputScenario::SelfReference => {
                vec![
                    original_id
                        .clone()
                        .expect("original must exist for SelfReference"),
                ]
            }
            InvalidInputScenario::NonExistent | InvalidInputScenario::CrossWorkspace => vec![],
        };

        let err = TicketBuilder::new(&store, target_ws)
            .title("New")
            .prereqs(&prereqs)
            .supersede(supersede_id)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains(expected_error),
            "Case '{}': expected error containing '{}', got: {err}",
            case.name,
            expected_error
        );
    }
}

#[tokio::test]
async fn test_supersede_tool() {
    crate::util::test::init_test_stores().await;

    let store = crate::board::BOARD.get().unwrap();
    let ws = test_ws("/tmp/test_ws_supersede_tool");

    // Create old ticket
    let old_id = make_ticket(store, &ws, "Old", TicketPhase::Backlog).await;

    let tool = crate::tools::CreateTicketTool::new("test");
    let args = serde_json::json!({
        "title": "Refined",
        "description": "refined desc",
        "supersede": old_id,
    });
    let result = tool.execute(&ws, args).await.expect("execute");
    assert!(
        result.contains("Superseded"),
        "Output should say Superseded: {result}"
    );
    assert!(
        result.contains(&old_id),
        "Output should mention old ID: {result}"
    );

    // Verify old is cancelled
    let old = expect_ticket(store, &old_id).await;
    assert_superseded_ticket(&old);
}

#[tokio::test]
async fn test_transactional_triple_write() {
    for should_commit in [false, true] {
        // Exercise the full pattern used by finalize_commit_and_transition:
        // all three _tx writes (set_commit_info_tx, transition_to_tx,
        // add_comment_tx) in one transaction → commit → all visible
        // (or rollback → none persist).  This is the sole transactional
        // test for set_commit_info_tx (its standalone test was removed
        // as subsumed); the commit_hash, lines_added, and lines_removed
        // assertions below verify its behavior under both commit and
        // rollback, complementing the non-transactional coverage in
        // test_ticket_roundtrip_all_fields.
        // Now delegates to the real production method BoardStore::finalize_done_tx.
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");
        let id = make_ticket(&store, &ws, "Test", TicketPhase::QaPassed).await;

        let tx = store.conn.begin_tx().await.unwrap();
        BoardStore::finalize_done_tx(
            &tx,
            &id,
            "abcdef0123456789abcdef0123456789abcd0123",
            10,
            5,
            "triple write comment",
            TicketPhase::QaPassed,
        )
        .await
        .unwrap();

        let label = if should_commit { "commit" } else { "rollback" };
        if should_commit {
            tx.commit().await.unwrap();
        } else {
            tx.rollback().await.unwrap();
        }

        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        let comments = store.get_comments(&id).await.expect("get comments");
        if should_commit {
            // All three changes should be visible.
            assert_eq!(
                ticket.commit_hash.as_deref(),
                Some("abcdef0123456789abcdef0123456789abcd0123"),
                "({label}) commit_hash",
            );
            assert_eq!(ticket.lines_added, Some(10), "({label}) lines_added");
            assert_eq!(ticket.lines_removed, Some(5), "({label}) lines_removed");
            assert_eq!(ticket.phase, TicketPhase::Done, "({label}) phase");
            assert_eq!(comments.len(), 1, "({label}) comments.len");
            assert_eq!(
                comments[0].content, "triple write comment",
                "({label}) comment content"
            );
        } else {
            // None of the three changes should be visible.
            assert_eq!(
                ticket.commit_hash, None,
                "({label}) commit_hash after rollback"
            );
            assert_eq!(
                ticket.lines_added, None,
                "({label}) lines_added after rollback"
            );
            assert_eq!(
                ticket.lines_removed, None,
                "({label}) lines_removed after rollback"
            );
            assert_eq!(
                ticket.phase,
                TicketPhase::QaPassed,
                "({label}) phase after rollback",
            );
            assert_eq!(comments.len(), 0, "({label}) comments.len after rollback");
        }
    }
}

// ── parse_prereqs unit tests ──

#[test]
fn test_parse_prereqs() {
    // ── Valid JSON cases ──
    let valid: &[(&str, &[&str])] = &[
        ("[]", &[] as &[&str]),
        (r#"["a","b","c"]"#, &["a", "b", "c"]),
    ];
    for (input, expected) in valid {
        let got = parse_prereqs(input).expect("should parse valid JSON");
        assert_eq!(got, *expected, "input: {input:?}");
    }

    // ── Invalid / corrupt JSON cases ──
    let invalid: &[&str] = &["", "not valid json {{{", r#"{"key":"value"}"#, "[1, 2, 3]"];
    for input in invalid {
        let err = parse_prereqs(input).unwrap_err();
        assert!(
            err.to_string().contains("Corrupt prerequisites JSON"),
            "input {input:?}: expected 'Corrupt prerequisites JSON' error, got: {err}",
        );
    }

    // ── Long ASCII input (>200 bytes) — preview truncated with ellipsis ──
    let long = format!(r#""{}...""#, "x".repeat(500));
    let msg = parse_prereqs(&long).unwrap_err().to_string();
    assert!(
        msg.contains('…'),
        "long input should produce truncated preview: {msg}"
    );
    assert!(
        msg.len() < 500,
        "truncated message should be <500 chars, got len={}",
        msg.len()
    );

    // ── Multi-byte character straddling byte 200 — no panic on truncation ──
    // Without floor_char_boundary, `&raw[..200]` would panic on the mid-char slice.
    let raw = format!("{}éééééééééémore", "x".repeat(199));
    assert!(raw.len() > 200, "need raw longer than 200 chars");
    // Verify byte 200 is indeed within a multi-byte character (not a boundary).
    assert!(
        !raw.is_char_boundary(200),
        "byte 200 must be mid-character for this test to be meaningful"
    );
    let msg = parse_prereqs(&raw).unwrap_err().to_string();
    assert!(
        msg.contains('…'),
        "multi-byte input should produce truncated preview: {msg}"
    );
    assert!(
        msg.len() < raw.len() + 50,
        "message too long after truncation: len={}, raw.len()={}",
        msg.len(),
        raw.len()
    );
    assert!(
        msg.contains("Corrupt prerequisites JSON"),
        "should mention corrupt JSON: {msg}"
    );
}

// ── Integration test: corrupt prerequisites in the database ──

#[tokio::test]
async fn corrupt_prerequisites_causes_query_errors() {
    let (store, _tmp, id) = setup().await;

    // Directly corrupt the prerequisites column via raw SQL
    store
        .conn
        .execute(
            "UPDATE tickets SET prerequisites = ?1 WHERE id = ?2",
            crate::turso::params!["{not valid json}", id.clone()],
        )
        .await
        .expect("corrupt update");

    // get_ticket should fail when prerequisites are corrupt
    let result = store.get_ticket(&id).await;
    assert!(
        result.is_err(),
        "get_ticket should fail when prerequisites are corrupt"
    );
    let err = result.unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Corrupt prerequisites JSON"),
        "error should mention corrupt JSON: {msg}"
    );
    assert!(
        msg.contains(&id),
        "error should include ticket ID {id}: {msg}"
    );

    // list_all_tickets should also fail entirely
    let result = store.list_all_tickets(Some("ws"), None).await;
    assert!(
        result.is_err(),
        "list_all_tickets should fail when any ticket has corrupt prerequisites"
    );
    let err = result.unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Corrupt prerequisites JSON"),
        "list_all_tickets error should mention corrupt JSON: {msg}"
    );
    assert!(
        msg.contains(&id),
        "list_all_tickets error should include ticket ID {id}: {msg}"
    );
}

// ── claim_diagnostics tests ──

/// Table-driven tests for `claim_diagnostics` covering success,
/// pre-assignment rejection, wrong-phase rejection, and idempotency.
#[tokio::test]
async fn test_claim_diagnostics() {
    enum Scenario {
        /// Ticket is unassigned and in InDiagnostics — claim should succeed.
        Success,
        /// Ticket is already assigned — claim should fail.
        AlreadyAssigned,
        /// Ticket is in a different phase — claim should fail.
        WrongPhase,
    }

    struct Case {
        name: &'static str,
        scenario: Scenario,
    }

    let cases = [
        Case {
            name: "unassigned in diagnostics succeeds",
            scenario: Scenario::Success,
        },
        Case {
            name: "already assigned fails",
            scenario: Scenario::AlreadyAssigned,
        },
        Case {
            name: "wrong phase fails",
            scenario: Scenario::WrongPhase,
        },
    ];

    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    for (i, case) in cases.iter().enumerate() {
        let title = format!("claim-{i}");
        let phase = if matches!(case.scenario, Scenario::WrongPhase) {
            TicketPhase::Backlog
        } else {
            TicketPhase::InDiagnostics
        };
        let id = make_ticket(&store, &ws, &title, phase).await;

        if matches!(case.scenario, Scenario::AlreadyAssigned) {
            store
                .set_assigned_to(&id, Some(DIAGNOSTICS_ROLE))
                .await
                .expect("set_assigned_to");
        }

        let claimed = store
            .claim_diagnostics(&id, DIAGNOSTICS_ROLE)
            .await
            .expect("claim_diagnostics");

        match case.scenario {
            Scenario::Success => {
                assert!(claimed, "Case '{}': expected claim to succeed", case.name);

                // Verify post-claim state.
                let ticket = crate::util::test::expect_ticket(&store, &id).await;
                assert_eq!(
                    ticket.assigned_to.as_deref(),
                    Some(DIAGNOSTICS_ROLE),
                    "Case '{}': assignee should be set",
                    case.name
                );
                assert_eq!(
                    ticket.phase,
                    TicketPhase::InDiagnostics,
                    "Case '{}': phase should remain InDiagnostics",
                    case.name
                );

                // Verify idempotency (second claim returns false).
                let second = store
                    .claim_diagnostics(&id, DIAGNOSTICS_ROLE)
                    .await
                    .expect("second claim");
                assert!(
                    !second,
                    "Case '{}': second claim should return false (idempotent)",
                    case.name
                );
            }
            Scenario::AlreadyAssigned | Scenario::WrongPhase => {
                assert!(!claimed, "Case '{}': expected claim to fail", case.name);
            }
        }
    }
}

// ── claim_sanitation tests ──

/// Table-driven tests for `claim_sanitation` covering success (QaPassed),
/// wrong-phase rejection, and assigned_to verification on successful claim.
#[tokio::test]
async fn test_claim_sanitation() {
    struct Case {
        name: &'static str,
        phase: TicketPhase,
        expected_claim: bool,
    }

    let cases = [
        Case {
            name: "qa_passed succeeds",
            phase: TicketPhase::QaPassed,
            expected_claim: true,
        },
        Case {
            name: "backlog (wrong phase) fails",
            phase: TicketPhase::Backlog,
            expected_claim: false,
        },
        Case {
            name: "in_development (wrong phase) fails",
            phase: TicketPhase::InDevelopment,
            expected_claim: false,
        },
    ];

    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    for (i, case) in cases.iter().enumerate() {
        let title = format!("san-claim-{i}");
        let id = make_ticket(&store, &ws, &title, case.phase).await;

        // Compute before the call (needed as parameter even for non-claim cases).
        let expected_key =
            crate::session::ticket_session_key(&id, crate::Role::Sanitation.as_str());

        let claimed = store
            .claim_sanitation(&id, &expected_key)
            .await
            .expect("claim_sanitation");
        assert_eq!(
            claimed, case.expected_claim,
            "Case '{}': unexpected claim result",
            case.name
        );

        if case.expected_claim {
            let ticket = crate::util::test::expect_ticket(&store, &id).await;
            assert_eq!(ticket.phase, TicketPhase::InSanitation);
            assert_eq!(
                ticket.assigned_to.as_deref(),
                Some(expected_key.as_str()),
                "Case '{}': assigned_to should be set to sanitation session key",
                case.name
            );
        }
    }
}

#[tokio::test]
async fn test_claim_sanitation_workspace_serialization() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // First ticket in QaPassed
    let first_id = make_ticket(&store, &ws, "First", TicketPhase::QaPassed).await;

    // Second ticket in QaPassed — same workspace
    let second_id = make_ticket(&store, &ws, "Second", TicketPhase::QaPassed).await;

    // Compute session keys before the calls (needed as parameters).
    let first_key = crate::session::ticket_session_key(&first_id, crate::Role::Sanitation.as_str());
    let second_key =
        crate::session::ticket_session_key(&second_id, crate::Role::Sanitation.as_str());

    // Claim the first — should succeed
    let first_claimed = store
        .claim_sanitation(&first_id, &first_key)
        .await
        .expect("first claim");
    assert!(first_claimed, "first claim should succeed");

    // Claim the second while first is in InSanitation — should fail (serialized)
    let second_claimed = store
        .claim_sanitation(&second_id, &second_key)
        .await
        .expect("second claim");
    assert!(
        !second_claimed,
        "second claim should be blocked while first ticket is in sanitation pipeline"
    );

    // Transition first ticket out of the sanitation pipeline entirely
    // (simulating the real flow: SanitationPassed → auto-commit → Done).
    // We transition directly to Done since SanitationPassed is also in the
    // blocked set, so moving to SanitationPassed alone wouldn't clear it.
    store
        .transition_to(&first_id, None, TicketPhase::Done, None)
        .await
        .expect("transition first to Done (clears sanitation pipeline)");

    // Now second claim should succeed
    let second_claimed_retry = store
        .claim_sanitation(&second_id, &second_key)
        .await
        .expect("second claim retry");
    assert!(
        second_claimed_retry,
        "second claim should succeed after pipeline clears"
    );
}

#[tokio::test]
async fn test_claim_sanitation_cross_workspace_serialization() {
    let (store, _tmp) = open_test_store().await;
    let ws_a = test_ws_named("/ws_a", "ws_a");
    let ws_b = test_ws_named("/ws_b", "ws_b");

    // One ticket in each workspace, both in QaPassed
    let id_a = make_ticket(&store, &ws_a, "Workspace A", TicketPhase::QaPassed).await;
    let id_b = make_ticket(&store, &ws_b, "Workspace B", TicketPhase::QaPassed).await;

    // Compute session keys before the calls (needed as parameters).
    let key_a = crate::session::ticket_session_key(&id_a, crate::Role::Sanitation.as_str());
    let key_b = crate::session::ticket_session_key(&id_b, crate::Role::Sanitation.as_str());

    // Both should succeed independently (different workspaces)
    let claimed_a = store
        .claim_sanitation(&id_a, &key_a)
        .await
        .expect("claim a");
    assert!(claimed_a, "workspace A claim should succeed");

    let claimed_b = store
        .claim_sanitation(&id_b, &key_b)
        .await
        .expect("claim b");
    assert!(
        claimed_b,
        "workspace B claim should succeed independently of workspace A"
    );

    let ticket_a = crate::util::test::expect_ticket(&store, &id_a).await;
    assert_eq!(ticket_a.phase, TicketPhase::InSanitation);

    let ticket_b = crate::util::test::expect_ticket(&store, &id_b).await;
    assert_eq!(ticket_b.phase, TicketPhase::InSanitation);
}

#[tokio::test]
async fn test_set_assigned_to_none() {
    // Successfully clear an assigned assignee
    let (store, _tmp, id) = setup().await;

    store
        .set_assigned_to(&id, Some(DIAGNOSTICS_ROLE))
        .await
        .expect("set_assigned_to");
    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert_eq!(ticket.assigned_to.as_deref(), Some(DIAGNOSTICS_ROLE));

    store
        .set_assigned_to(&id, None)
        .await
        .expect("set_assigned_to(None) should clear assignee");
    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert!(ticket.assigned_to.is_none(), "assigned_to should be NULL");

    // Idempotent: clearing an already-None assignee succeeds
    store
        .set_assigned_to(&id, None)
        .await
        .expect("second set_assigned_to(None) should also succeed");

    // Non-existent ticket fails
    let (store2, _tmp2) = open_test_store().await;
    let result = store2.set_assigned_to("nonexistent", None).await;
    assert!(
        result.is_err(),
        "set_assigned_to(None) on nonexistent ticket should fail"
    );
}

/// Round-trip test that exercises ALL column-index constants in
/// [`ticket_from_row`] by creating a ticket, setting every mutable field
/// via public API, then verifying every [`Ticket`] field (including
/// `pipeline_reservation` via its SQL `DEFAULT 0`) survives the
/// SELECT → `ticket_from_row` deserialization path.
///
/// Serves as a regression test for ticket deserialization — the
/// [`columns!`] macro ensures single-sourcing of [`TICKET_COLUMNS`]
/// and [`COL_TICKET_*`], so column-order drift between them is
/// structurally impossible. This test still exercises the full
/// `ticket_from_row` deserialization path, including manual
/// field-by-field extraction via `row.get::<Type>(COL_TICKET_*)`
/// and default-value handling.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn test_ticket_roundtrip_all_fields() {
    let (store, _tmp) = open_test_store().await;

    // Non-existent ticket returns None.
    let none = store.get_ticket("nonexistent").await.expect("get");
    assert!(none.is_none(), "non-existent ticket should return None");

    let ws = crate::workspace::test_ws_named("/test_ws", "test_workspace");

    // Create ticket with known values for every TICKET_COLUMNS position.
    let id = TicketBuilder::new(&store, &ws)
        .title("Roundtrip Title")
        .desc("Roundtrip description")
        .phase(TicketPhase::Backlog)
        .reporter("test_reporter")
        .create()
        .await
        .expect("create_ticket");

    // Read back BEFORE setting any mutable fields — verify fresh-ticket defaults
    // (None for assigned_to, commit_hash, lines_added, lines_removed; empty comments).
    let fresh = store
        .get_ticket(&id)
        .await
        .expect("get_ticket")
        .expect("ticket exists");
    assert_eq!(fresh.title, "Roundtrip Title", "fresh title");
    assert_eq!(
        fresh.description, "Roundtrip description",
        "fresh description"
    );
    assert_eq!(fresh.phase, TicketPhase::Backlog, "fresh phase");
    assert!(
        fresh.assigned_to.is_none(),
        "fresh ticket should have no assigned_to"
    );
    assert!(
        fresh.comments.is_empty(),
        "fresh ticket should have no comments"
    );
    assert!(
        fresh.commit_hash.is_none(),
        "fresh ticket should have no commit_hash"
    );
    assert!(
        fresh.lines_added.is_none(),
        "fresh ticket should have no lines_added"
    );
    assert!(
        fresh.lines_removed.is_none(),
        "fresh ticket should have no lines_removed"
    );
    assert_eq!(
        fresh.workspace_name, "test_workspace",
        "fresh workspace_name"
    );
    assert!(
        fresh.created_at.contains('T'),
        "fresh created_at should be RFC 3339: {}",
        fresh.created_at,
    );
    assert!(
        fresh.updated_at.contains('T'),
        "fresh updated_at should be RFC 3339: {}",
        fresh.updated_at,
    );
    assert_eq!(fresh.reporter, "test_reporter", "fresh reporter");
    assert!(
        fresh.prerequisites.is_empty(),
        "fresh prerequisites should be empty"
    );
    assert!(
        fresh.supersedes.is_none(),
        "fresh supersedes should be None"
    );
    assert!(
        fresh.superseded_by.is_none(),
        "fresh superseded_by should be None"
    );
    assert!(!fresh.is_archived, "fresh is_archived should be false");
    assert!(
        !fresh.pipeline_reservation,
        "fresh pipeline_reservation should be false"
    );

    // Set assigned_to (exercises COL_TICKET_ASSIGNED_TO with non-None value).
    store
        .set_assigned_to(&id, Some("test_assignee"))
        .await
        .expect("set_assigned_to");

    // Set commit_hash, lines_added, lines_removed with non-default values.
    let tx = store.conn.begin_tx().await.unwrap();
    BoardStore::set_commit_info_tx(&tx, &id, "abcdef0123456789abcdef0123456789abcd0123", 42, 7)
        .await
        .expect("set_commit_info_tx");
    tx.commit().await.unwrap();

    // Read back BEFORE archiving (which clears assigned_to).
    let ticket = store
        .get_ticket(&id)
        .await
        .expect("get_ticket")
        .expect("ticket exists");

    // ── Assert every Ticket field round-trips ──────────────────────
    assert_eq!(ticket.id, id, "id mismatch");
    assert_eq!(ticket.title, "Roundtrip Title", "title mismatch");
    assert_eq!(
        ticket.description, "Roundtrip description",
        "description mismatch",
    );
    assert_eq!(ticket.phase, TicketPhase::Backlog, "phase mismatch");
    assert_eq!(
        ticket.assigned_to.as_deref(),
        Some("test_assignee"),
        "assigned_to should round-trip",
    );
    assert_eq!(ticket.workspace_name, "test_workspace");
    // Timestamps are auto-generated RFC 3339 — validate format, not value.
    assert!(
        ticket.created_at.contains('T'),
        "created_at should be RFC 3339: {}",
        ticket.created_at,
    );
    assert!(
        ticket.updated_at.contains('T'),
        "updated_at should be RFC 3339: {}",
        ticket.updated_at,
    );
    assert!(ticket.comments.is_empty(), "no comments expected");
    assert!(
        ticket.prerequisites.is_empty(),
        "prerequisites should round-trip as empty",
    );
    assert_eq!(
        ticket.commit_hash.as_deref(),
        Some("abcdef0123456789abcdef0123456789abcd0123"),
        "commit_hash mismatch",
    );
    assert_eq!(ticket.lines_added, Some(42), "lines_added mismatch");
    assert_eq!(ticket.lines_removed, Some(7), "lines_removed mismatch");
    assert_eq!(ticket.reporter, "test_reporter", "reporter mismatch");
    // Fields not set remain at their defaults.
    assert!(
        ticket.supersedes.is_none(),
        "supersedes should be None for simple ticket",
    );
    assert!(
        ticket.superseded_by.is_none(),
        "superseded_by should be None for simple ticket",
    );
    assert!(
        !ticket.is_archived,
        "is_archived should be false before archiving",
    );
    assert!(
        !ticket.pipeline_reservation,
        "pipeline_reservation should be false for fresh ticket",
    );

    // ── Exercise is_archived bool deserialization ──────────────────
    // set_archived flips is_archived to 1 in SQL, which exercises the
    // conversion: row.get::<bool>()?.
    store.set_archived(&id).await.expect("set_archived");

    let archived = store
        .get_ticket(&id)
        .await
        .expect("get_ticket")
        .expect("ticket exists after archive");
    assert!(
        archived.is_archived,
        "is_archived should be true after set_archived"
    );
    assert!(
        archived.assigned_to.is_none(),
        "assigned_to should be cleared after archive",
    );
}

// ── Archived ticket search methods ──────────────────────────────────

/// Create an archived ticket with the given title in tests.
async fn create_archived_ticket(
    store: &super::BoardStore,
    title: &str,
    workspace_name: &str,
) -> String {
    let ws = test_ws(workspace_name);
    let id = make_ticket(store, &ws, title, crate::board::TicketPhase::Done).await;
    store.set_archived(&id).await.expect("set_archived");
    id
}

#[tokio::test]
async fn test_search_archived_by_fts_finds_matching_title() {
    let (store, _tmp) = open_test_store().await;
    let id = create_archived_ticket(&store, "Fix network timeout bug", "ws1").await;

    let results = store
        .search_archived_by_fts("network timeout", 10)
        .await
        .expect("FTS search");
    assert!(!results.is_empty(), "should find the ticket");
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        ids.contains(&id.as_str()),
        "result should contain our ticket"
    );
}

#[tokio::test]
async fn test_search_archived_by_fts_excludes_non_archived() {
    let (store, _tmp) = open_test_store().await;
    // Create a non-archived ticket — should not appear in archived search
    let ws = test_ws("ws2");
    make_ticket(
        &store,
        &ws,
        "Still active",
        crate::board::TicketPhase::Backlog,
    )
    .await;

    let results = store
        .search_archived_by_fts("active", 10)
        .await
        .expect("FTS search");
    assert!(results.is_empty(), "non-archived ticket should not appear");
}

#[tokio::test]
async fn test_search_archived_by_fts_punctuation_only() {
    let (store, _tmp) = open_test_store().await;
    let results = store
        .search_archived_by_fts("!@#$%", 10)
        .await
        .expect("FTS search");
    assert!(
        results.is_empty(),
        "query with only punctuation should produce no matches"
    );
}

/// Basic field layout of `detailed_display`: fields present, negative
/// assertions for absent fields, and "(no comments)" when empty.
#[tokio::test]
async fn test_detailed_display_basic() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/test-workspace", "test-ws");

    let prereq_id = make_ticket(&store, &ws, "Prereq", TicketPhase::Backlog).await;

    let id = TicketBuilder::new(&store, &ws)
        .title("Display Test Ticket")
        .desc("A description for testing")
        .phase(TicketPhase::InDevelopment)
        .prereqs(std::slice::from_ref(&prereq_id))
        .reporter("manager")
        .create()
        .await
        .expect("create");

    let ticket = expect_ticket(&store, &id).await;
    let display = ticket.detailed_display();

    assert!(
        display.contains(&format!("Ticket: {id}")),
        "should contain ticket id"
    );
    assert!(
        display.contains("Title: Display Test Ticket"),
        "should contain title"
    );
    assert!(
        display.contains("Description: A description for testing"),
        "should contain description"
    );
    assert!(
        display.contains("Phase: in_development"),
        "should use snake_case phase"
    );
    assert!(
        display.contains("Reporter: manager"),
        "should contain reporter"
    );
    assert!(
        display.contains("Workspace: test-ws"),
        "should contain workspace"
    );
    assert!(
        display.contains("Created:"),
        "should contain created timestamp"
    );
    assert!(
        display.contains("Updated:"),
        "should contain updated timestamp"
    );
    assert!(
        display.contains(&format!("Prerequisites: {prereq_id}")),
        "should show prerequisites"
    );
    assert!(
        display.contains("Comments:"),
        "should have comments section"
    );
    assert!(display.contains("(no comments)"), "should show no comments");

    // Fields that should NOT appear when unset
    assert!(
        !display.contains("Supersedes:"),
        "no supersedes when not set"
    );
    assert!(
        !display.contains("Superseded by:"),
        "no superseded_by when not set"
    );
    assert!(
        !display.contains("Archived:"),
        "no archived line when false"
    );
    assert!(
        !display.contains("assigned_to:"),
        "assigned_to should not be displayed"
    );
    assert!(
        !display.contains("commit_hash:"),
        "commit_hash should not be displayed"
    );
    assert!(
        !display.contains("lines_added:"),
        "lines_added should not be displayed"
    );
    assert!(
        !display.contains("lines_removed:"),
        "lines_removed should not be displayed"
    );
}

/// `detailed_display` with comments (role labels, content) and multiple
/// prerequisites joined by comma+space.
#[tokio::test]
async fn test_detailed_display_with_content() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/test-workspace", "test-ws");

    // ── Comment formatting: two comments with different roles ──

    let id = make_ticket(&store, &ws, "Comment Test", TicketPhase::Backlog).await;

    store
        .add_comment(&id, Role::Analyst.as_str(), "First comment")
        .await
        .expect("add_comment");
    store
        .add_comment(&id, Role::Reviewer.as_str(), "Second comment")
        .await
        .expect("add_comment");

    let ticket = expect_ticket(&store, &id).await;
    let display = ticket.detailed_display();

    assert!(
        display.contains("Comments:"),
        "should have comments section"
    );
    assert!(display.contains("[analyst]"), "should show analyst role");
    assert!(display.contains("[reviewer]"), "should show reviewer role");
    assert!(
        display.contains("First comment"),
        "should show first comment"
    );
    assert!(
        display.contains("Second comment"),
        "should show second comment"
    );
    assert!(
        !display.contains("(no comments)"),
        "should not say 'no comments' when comments exist"
    );

    // ── Multiple prerequisites: all three joined by comma+space ──

    let pre_a = make_ticket(&store, &ws, "Pre-A", TicketPhase::Backlog).await;
    let pre_b = make_ticket(&store, &ws, "Pre-B", TicketPhase::Backlog).await;
    let pre_c = make_ticket(&store, &ws, "Pre-C", TicketPhase::Backlog).await;

    let multi_id = TicketBuilder::new(&store, &ws)
        .title("Multi prereq")
        .prereqs(&[pre_a.clone(), pre_b.clone(), pre_c.clone()])
        .create()
        .await
        .expect("create");

    let ticket = expect_ticket(&store, &multi_id).await;
    let display = ticket.detailed_display();

    assert!(
        display.contains(&format!("Prerequisites: {pre_a}, {pre_b}, {pre_c}")),
        "should show all prerequisites joined with comma+space"
    );
}

/// `detailed_display` for supersedes chains: new ticket shows Supersedes,
/// old ticket shows Superseded by + Archived.
#[tokio::test]
async fn test_detailed_display_supersedes_chain() {
    init_test_stores().await;
    let store = crate::board::BOARD.get().unwrap();
    let ws = test_ws_named("/ws", "ws");

    // Create an old ticket first
    let old_id = make_ticket(store, &ws, "Old ticket", TicketPhase::Backlog).await;

    // Supersede it — new ticket gets supersedes = old_id, old ticket gets
    // superseded_by = new_id and is archived.
    let new_id = TicketBuilder::new(store, &ws)
        .title("New ticket")
        .desc("new desc")
        .supersede(&old_id)
        .await
        .expect("supersede");

    // Check the new ticket shows Supersedes
    let new_ticket = expect_ticket(store, &new_id).await;
    let new_display = new_ticket.detailed_display();
    assert!(
        new_display.contains(&format!("Supersedes: {old_id}")),
        "new ticket should show Supersedes: old_id"
    );

    // Check the old ticket shows Superseded by + Archived
    let old_ticket = expect_ticket(store, &old_id).await;
    let old_display = old_ticket.detailed_display();
    assert!(
        old_display.contains(&format!("Superseded by: {new_id}")),
        "old ticket should show Superseded by: new_id"
    );
    assert!(
        old_display.contains("Archived: yes"),
        "old ticket should be archived"
    );
}

#[tokio::test]
async fn test_list_archived_with_embeddings_returns_deserialized() {
    let (store, _tmp) = open_test_store().await;

    // Empty DB returns empty
    {
        let candidates = store.list_archived_with_embeddings().await.expect("list");
        assert!(candidates.is_empty(), "no tickets at all");
    }

    let ws = test_ws("ws");

    // Create a ticket with a known embedding blob (two small f32s)
    let embedding: Vec<f32> = vec![1.0, 2.0];
    let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

    let id = TicketBuilder::new(&store, &ws)
        .title("Embedded ticket")
        .phase(crate::board::TicketPhase::Done)
        .embedding(&blob)
        .create()
        .await
        .expect("create_ticket with embedding");
    store.set_archived(&id).await.expect("archive");

    let candidates = store.list_archived_with_embeddings().await.expect("list");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0, id);
    assert_eq!(candidates[0].1, vec![1.0, 2.0]);
}
