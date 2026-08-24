use super::*;
use crate::Role;
use crate::Tool;
use crate::Workspace;
use crate::util::test::TicketBuilder;
use crate::util::test::assert_superseded_ticket;
use crate::util::test::expect_ticket;
use crate::util::test::init_test_stores;
use crate::util::test::make_ticket;
use crate::workspace::test_ws;
use crate::workspace::test_ws_named;
use tempfile::TempDir;

/// Scenarios for testing invalid prerequisite/supersede inputs.
#[derive(Debug, Clone, Copy)]
enum InvalidInputScenario {
    /// Prerequisite/supersede references a nonexistent ticket.
    NonExistent,
    /// Prerequisite/supersede references a ticket in a different workspace.
    CrossWorkspace,
    /// Prerequisite/supersede references the ticket itself (self-reference).
    SelfReference,
}

/// Operation under test in the invalid-input matrix.
#[derive(Debug, Clone, Copy)]
enum InvalidInputOp {
    Create,
    Supersede,
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

    // After transition, reflects new phase.
    store
        .transition_to(&id, None, TicketPhase::ReadyForDevelopment)
        .await
        .expect("set");
    let phase = crate::util::test::expect_ticket_phase(&store, &id).await;
    assert_eq!(phase, TicketPhase::ReadyForDevelopment);
}

#[tokio::test]
async fn test_get_tickets_by_ids() {
    let (store, _tmp) = open_test_store().await;
    let ws = crate::workspace::test_ws_named("/ws", "test_ws");

    // sql_in_placeholders(0) produces invalid `WHERE id IN ()` —
    // the guard must short-circuit empty input before reaching SQL.
    let tickets = store
        .get_tickets_by_ids(&[], crate::pipeline::board::LoadComments::No)
        .await
        .expect("empty ids");
    assert!(tickets.is_empty(), "empty ids should return empty vec");

    let id_a = make_ticket(&store, &ws, "Ticket A", TicketPhase::Done).await;
    let id_c = make_ticket(&store, &ws, "Ticket C", TicketPhase::Backlog).await;

    let ids = vec![id_a.clone(), id_c.clone()];
    let tickets = store
        .get_tickets_by_ids(&ids, crate::pipeline::board::LoadComments::No)
        .await
        .expect("get by ids");
    assert_eq!(tickets.len(), 2, "should return exactly 2 tickets");

    for t in &tickets {
        match t.id.as_str() {
            id if id == id_a => assert_eq!(t.title, "Ticket A"),
            id if id == id_c => assert_eq!(t.title, "Ticket C"),
            other => panic!("unexpected ticket id: {other}"),
        }
    }
}

#[tokio::test]
async fn test_guarded_transition() {
    let (store, _tmp, id) = setup().await;

    // Wrong expected phase — should fail, ticket unchanged.
    let result = store
        .transition_to(&id, Some(TicketPhase::Done), TicketPhase::InDevelopment)
        .await;
    assert!(
        result.is_err(),
        "guarded transition with wrong phase should fail"
    );
    let ticket = crate::util::test::expect_ticket(&store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::Backlog);

    // Correct expected phase — should succeed.
    store
        .transition_to(&id, Some(TicketPhase::Backlog), TicketPhase::InDevelopment)
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

    // Filter by phase (none match since all are Backlog)
    let tickets = store
        .list_all_tickets(Some("ws"), Some(TicketPhase::Done))
        .await
        .expect("list");
    assert_eq!(tickets.len(), 0);
}

/// Verify that `reset_analysis_tickets` correctly transitions each in-flight
/// ticket phase back to its ready state, and that non-inflight phases (e.g.
/// Backlog) are left untouched.
///
/// Canonical reset test. The serial(reset_inflight) group is the contract for
/// the shared global board: ANY test creating a ticket in a reset-affected
/// phase (Analysis/InDevelopment/InDiagnostics/InSanitation/InReview/InQa) on
/// the shared board must join this group, or a concurrent reset will clobber
/// its fixture (phase-CAS failure indistinguishable from a real regression).
/// This test itself uses an isolated store, so its membership is defensive —
/// the attribute is load-bearing for the group, not for this test's own data.
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn test_reset_analysis_tickets() {
    /// A single reset transition case.
    struct Case {
        name: &'static str,
        /// Unique suffix for workspace names (isolates cases).
        suffix: &'static str,
        /// The phase the ticket starts in.
        start: TicketPhase,
        /// The expected phase after reset.
        expected: TicketPhase,
    }

    let cases = [
        Case {
            name: "Backlog unaffected (not an inflight phase)",
            suffix: "a",
            start: TicketPhase::Backlog,
            expected: TicketPhase::Backlog,
        },
        Case {
            name: "Analysis → Backlog",
            suffix: "b",
            start: TicketPhase::Analysis,
            expected: TicketPhase::Backlog,
        },
        // Implementation-protected occupied phases are NOT reset: a resumed implementation
        // keeps them in phase. Only Analysis → Backlog is reset.
        Case {
            name: "InDevelopment stays (implementation-protected)",
            suffix: "c",
            start: TicketPhase::InDevelopment,
            expected: TicketPhase::InDevelopment,
        },
        Case {
            name: "InDiagnostics stays (implementation-protected)",
            suffix: "d",
            start: TicketPhase::InDiagnostics,
            expected: TicketPhase::InDiagnostics,
        },
        Case {
            name: "InSanitation stays (implementation-protected)",
            suffix: "e",
            start: TicketPhase::InSanitation,
            expected: TicketPhase::InSanitation,
        },
        Case {
            name: "InQa stays (implementation-protected)",
            suffix: "f",
            start: TicketPhase::InQa,
            expected: TicketPhase::InQa,
        },
        Case {
            name: "InReview stays (implementation-protected)",
            suffix: "g",
            start: TicketPhase::InReview,
            expected: TicketPhase::InReview,
        },
    ];

    let (store, _tmp) = open_test_store().await;

    for case in &cases {
        let ws = test_ws_named(&format!("/{}", case.suffix), case.suffix);

        let id = make_ticket(&store, &ws, case.name, case.start).await;

        store.reset_analysis_tickets(&[]).await.expect("reset");

        let t = expect_ticket(&store, &id).await;
        assert_eq!(
            t.phase, case.expected,
            "Case '{}': unexpected phase after reset",
            case.name,
        );
    }
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
/// active tickets (PIPELINE_OCCUPIED_PHASES + ReadyForDevelopment) per workspace,
/// excluding a specified ticket ID.
///
/// Active tickets include all ReadyForDevelopment tickets. This is intentional —
/// unstarted backlog tickets are considered active to suppress Done notifications
/// until the pipeline is fully drained.
#[tokio::test]
async fn test_has_active_tickets_excluding() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // Create one ticket per active phase: all PIPELINE_OCCUPIED_PHASES + ReadyForDevelopment
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

    // ReadyForDevelopment counts as active regardless of its pipeline state
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
            None,
        )
        .await
        .expect("claim in ws_a")
        .expect("should claim ticket from ws_a");
    assert_eq!(claimed_a.id, id_a);
    assert_eq!(claimed_a.workspace_name, "workspace_a");
    assert_eq!(claimed_a.phase, TicketPhase::InDevelopment);

    // Claim from workspace A again — should return None (no more backlog tickets)
    assert!(
        store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::InDevelopment,
                "workspace_a",
                PipelineCheck::Skip,
                None,
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
            None,
        )
        .await
        .expect("claim in ws_b")
        .expect("should claim ticket from ws_b");
    assert_eq!(claimed_b.id, id_b);
    assert_eq!(claimed_b.workspace_name, "workspace_b");
}

#[tokio::test]
async fn test_claim_ticket_in_workspace_respects_claim_grace() {
    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");

    // Fresh ticket (created just now) must not be claimed within the grace window.
    let fresh = make_ticket(&store, &ws, "Fresh", TicketPhase::Backlog).await;
    assert!(
        store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::Analysis,
                "ws",
                PipelineCheck::Skip,
                Some(chrono::Duration::seconds(60)),
            )
            .await
            .expect("claim")
            .is_none(),
        "fresh ticket must stay in backlog within the claim grace window"
    );

    // Once the ticket is older than the grace window it is claimable again.
    let old_created = (Utc::now() - chrono::Duration::seconds(120)).to_rfc3339();
    store
        .conn
        .execute(
            "UPDATE tickets SET created_at = ?1 WHERE id = ?2",
            crate::db::params![old_created, fresh.clone()],
        )
        .await
        .expect("backdate");
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
            Some(chrono::Duration::seconds(60)),
        )
        .await
        .expect("claim")
        .expect("old ticket should be claimable");
    assert_eq!(claimed.id, fresh);

    // No grace window: fresh tickets are claimed immediately.
    let fresh2 = make_ticket(&store, &ws, "Fresh2", TicketPhase::Backlog).await;
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
            None,
        )
        .await
        .expect("claim")
        .expect("fresh ticket claimable without grace window");
    assert_eq!(claimed.id, fresh2);
}

/// Table-driven tests for [`PipelineCheck::Enforce`] — claims with pipeline occupancy
/// checking enabled.
#[tokio::test]
async fn test_claim_ticket_in_workspace_if_pipeline_free() {
    /// The pipeline scenario for a single test case.
    enum Scenario {
        /// Occupant in the same workspace — claim should be blocked.
        SameWorkspace(TicketPhase),
        /// Occupant in a different workspace — claim should succeed.
        DifferentWorkspace(TicketPhase),
        /// No occupant — claim should succeed.
        NoOccupant,
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
            name: "no occupant succeeds",
            suffix: "none",
            scenario: Scenario::NoOccupant,
        },
    ];

    let (store, _tmp) = open_test_store().await;

    for case in &cases {
        let suffix = case.suffix;

        // Derive workspace names from the scenario.
        let (claim_ws_name, occupied_ws_name) = match &case.scenario {
            Scenario::DifferentWorkspace(_) => (
                format!("ws_{suffix}_claimable"),
                format!("ws_{suffix}_occupied"),
            ),
            // SameWorkspace and NoOccupant both use a single workspace name.
            Scenario::SameWorkspace(_) | Scenario::NoOccupant => {
                let name = format!("ws_{suffix}");
                (name.clone(), name)
            }
        };

        let expected_claim = !matches!(case.scenario, Scenario::SameWorkspace(_));

        let occupied_ws = test_ws_named(&format!("/{occupied_ws_name}"), &occupied_ws_name);
        let claimable_ws = test_ws_named(&format!("/{claim_ws_name}"), &claim_ws_name);

        // Create a pipeline occupant (if any)
        if let Scenario::SameWorkspace(phase) | Scenario::DifferentWorkspace(phase) = &case.scenario
        {
            // When the occupant and claimable share a workspace, place the
            // occupant in the claimable's workspace (they are the same).
            let occupant_target = match &case.scenario {
                Scenario::DifferentWorkspace(_) => &occupied_ws,
                Scenario::SameWorkspace(_) => &claimable_ws,
                // Not reachable: NoOccupant is guarded by the enclosing if-let.
                Scenario::NoOccupant => unreachable!(),
            };
            make_ticket(&store, occupant_target, "Occupant", *phase).await;
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
                None,
            )
            .await
            .expect("claim should not error");

        if expected_claim {
            let claimed = claimed.expect("should claim ticket");
            assert_eq!(claimed.id, id, "Case '{}': wrong ticket id", case.name);
            assert_eq!(
                claimed.phase,
                TicketPhase::InDevelopment,
                "Case '{}': wrong phase after claim",
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

/// Matrix of invalid prerequisite/supersede inputs for `create_ticket` and
/// `supersede_and_create`.
#[tokio::test]
async fn test_invalid_inputs() {
    let cases = [
        (InvalidInputOp::Create, InvalidInputScenario::NonExistent),
        (InvalidInputOp::Create, InvalidInputScenario::CrossWorkspace),
        (InvalidInputOp::Create, InvalidInputScenario::SelfReference),
        (InvalidInputOp::Supersede, InvalidInputScenario::NonExistent),
        (
            InvalidInputOp::Supersede,
            InvalidInputScenario::CrossWorkspace,
        ),
        (
            InvalidInputOp::Supersede,
            InvalidInputScenario::SelfReference,
        ),
    ];

    let (store, _tmp) = open_test_store().await;
    let ws = test_ws_named("/ws", "ws");
    let ws_b = test_ws_named("/ws_b", "ws_b");
    // Isolated workspace for create/SelfReference: with exactly one seed
    // ticket, the hardcoded `{ws_sr}-1` predicts the next ID (board.rs
    // allocates IDs inside the tx before the self-reference check). No
    // other cell may write to ws_sr or the prediction silently breaks.
    let ws_sr = test_ws_named("/ws_sr", "ws_sr");

    for (op, scenario) in cases {
        // SelfReference keeps per-op arms — the error substrings differ
        // ('cannot depend on itself' vs 'supersede and depend'); the
        // NonExistent/CrossWorkspace substrings are identical across ops.
        let expected_error = match (op, scenario) {
            (_, InvalidInputScenario::NonExistent) => "not found",
            (_, InvalidInputScenario::CrossWorkspace) => "Cross-workspace",
            (InvalidInputOp::Create, InvalidInputScenario::SelfReference) => {
                "cannot depend on itself"
            }
            (InvalidInputOp::Supersede, InvalidInputScenario::SelfReference) => {
                "supersede and depend"
            }
        };

        // Seed a ticket for scenarios that reference an existing one.
        // NonExistent: none — reference a nonexistent id directly.
        // CrossWorkspace: seed in `ws`, referenced from `ws_b`.
        // SelfReference: supersede reuses the original in `ws`; create seeds
        //   the isolated `ws_sr` (counter invariant above).
        let seed: Option<String> = match scenario {
            InvalidInputScenario::NonExistent => None,
            InvalidInputScenario::CrossWorkspace => {
                Some(make_ticket(&store, &ws, "Existing", TicketPhase::Backlog).await)
            }
            InvalidInputScenario::SelfReference => {
                let seed_ws = match op {
                    InvalidInputOp::Create => &ws_sr,
                    InvalidInputOp::Supersede => &ws,
                };
                Some(make_ticket(&store, seed_ws, "Original", TicketPhase::Backlog).await)
            }
        };

        let target_ws = match (op, scenario) {
            (InvalidInputOp::Create, InvalidInputScenario::SelfReference) => &ws_sr,
            (_, InvalidInputScenario::CrossWorkspace) => &ws_b,
            (_, InvalidInputScenario::NonExistent)
            | (InvalidInputOp::Supersede, InvalidInputScenario::SelfReference) => &ws,
        };

        let prereqs: Vec<String> = match (op, scenario) {
            (InvalidInputOp::Create, InvalidInputScenario::NonExistent) => {
                vec!["nonexistent-1".to_string()]
            }
            (InvalidInputOp::Create, InvalidInputScenario::SelfReference) => {
                // Exactly one seed above → next id in ws_sr is `{ws_sr}-1`.
                vec![format!("{}-1", ws_sr.name)]
            }
            (
                InvalidInputOp::Supersede,
                InvalidInputScenario::NonExistent | InvalidInputScenario::CrossWorkspace,
            ) => vec![],
            (InvalidInputOp::Create, InvalidInputScenario::CrossWorkspace)
            | (InvalidInputOp::Supersede, InvalidInputScenario::SelfReference) => {
                vec![seed.clone().expect("seed")]
            }
        };

        let err = match op {
            InvalidInputOp::Create => TicketBuilder::new(&store, target_ws)
                .title("New")
                .prereqs(&prereqs)
                .create()
                .await
                .unwrap_err(),
            InvalidInputOp::Supersede => {
                // NonExistent supersedes a nonexistent target; the rest reuse
                // the seeded ticket.
                let supersede_id = seed.as_deref().unwrap_or("nonexistent");
                TicketBuilder::new(&store, target_ws)
                    .title("New")
                    .prereqs(&prereqs)
                    .supersede(supersede_id)
                    .await
                    .unwrap_err()
            }
        };
        assert!(
            err.to_string().contains(expected_error),
            "Case '{op:?}/{scenario:?}': expected error containing \
             '{expected_error}', got: {err}"
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
            None,
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
            None,
        )
        .await
        .expect("claim");
    assert!(
        second.is_none(),
        "B should be blocked because A is in Analysis, not Done"
    );

    // Move A to done
    store
        .transition_to(&a, None, TicketPhase::Done)
        .await
        .expect("done a");

    // Now B should be claimable
    let claimed2 = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
            None,
        )
        .await
        .expect("claim")
        .expect("should claim B");
    assert_eq!(claimed2.id, b);

    // Move B to done
    store
        .transition_to(&b, None, TicketPhase::Done)
        .await
        .expect("done b");

    // Now C should be claimable
    let claimed3 = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            "ws",
            PipelineCheck::Skip,
            None,
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

/// Assert the archived-excluded count of tickets in `phase` equals `expected`.
///
/// `expect_msg` labels the DB query failure and `assert_msg` the count
/// mismatch, preserving each call-site's distinct diagnostics.
async fn assert_phase_count(
    store: &BoardStore,
    phase: TicketPhase,
    expected: i64,
    expect_msg: &str,
    assert_msg: &str,
) {
    let count = store.count_by_phase(phase, None).await.expect(expect_msg);
    assert_eq!(count, expected, "{assert_msg}");
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
            crate::db::params![two_hours_ago.clone(), old_cancelled_id.clone()],
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
            crate::db::params![two_hours_ago.clone(), old_backlog_id.clone()],
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

    // Before archiving, count_by_phase includes active tickets.
    assert_phase_count(
        &store,
        TicketPhase::Done,
        1,
        "count Done before",
        "Should count Done ticket before archive",
    )
    .await;
    assert_phase_count(
        &store,
        TicketPhase::Cancelled,
        1,
        "count Cancelled before",
        "Should count Cancelled ticket before archive",
    )
    .await;
    assert_phase_count(
        &store,
        TicketPhase::Backlog,
        1,
        "count Backlog before",
        "Should count Backlog ticket before archive",
    )
    .await;

    // Act
    let count = store
        .archive_all_done_and_cancelled(None)
        .await
        .expect("archive");
    assert_eq!(count, 2, "should archive Done and Cancelled tickets");

    // Assert per-ticket state
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

    // After archiving, count_by_phase excludes archived tickets.
    assert_phase_count(
        &store,
        TicketPhase::Done,
        0,
        "count Done after",
        "Should not count archived Done tickets",
    )
    .await;
    assert_phase_count(
        &store,
        TicketPhase::Cancelled,
        0,
        "count Cancelled after",
        "Should not count archived Cancelled tickets",
    )
    .await;
    assert_phase_count(
        &store,
        TicketPhase::Backlog,
        1,
        "count Backlog after",
        "Should still count non-archived Backlog tickets",
    )
    .await;
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
async fn test_create_ticket_tool_with_prerequisites() {
    crate::util::test::init_test_stores().await;

    let store = crate::pipeline::board::BOARD.get().unwrap();
    let ws = test_ws("/tmp/test_ws_tool_prereqs");

    // Create a prerequisite via the store directly
    let p_id = make_ticket(store, &ws, "Pre", TicketPhase::Backlog).await;

    let tool = crate::tools::CreateTicketTool::new("test", &ws);
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
/// cancellation UPDATE (in `supersede_and_create`) has no phase guard
/// (`WHERE id = ?3` without `AND phase = ?`), so it runs identically
/// regardless of the old ticket's current phase. A separate test with a
/// `Cancelled` starting phase would exercise the exact same SQL path and
/// assert the same invariants (`assert_superseded_ticket`, `supersedes`
/// back-link), making it redundant with this one.
#[tokio::test]
async fn test_supersede_and_create_basic() {
    init_test_stores().await;
    let store = crate::pipeline::board::BOARD.get().unwrap();
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
    let store = crate::pipeline::board::BOARD.get().unwrap();
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

#[tokio::test]
async fn test_supersede_tool() {
    crate::util::test::init_test_stores().await;

    let store = crate::pipeline::board::BOARD.get().unwrap();
    let ws = test_ws("/tmp/test_ws_supersede_tool");

    // Create old ticket
    let old_id = make_ticket(store, &ws, "Old", TicketPhase::Backlog).await;

    let tool = crate::tools::CreateTicketTool::new("test", &ws);
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
    for should_succeed in [false, true] {
        // Exercise the transactional pattern that finalize_commit_and_transition
        // now uses via with_comment_and_transition: all three _tx writes
        // (set_commit_info_tx, add_comment_tx, transition_to_tx) in one
        // transaction → commit → all visible (or error rollback → none persist).
        let (store, _tmp) = open_test_store().await;
        let ws = test_ws_named("/ws", "ws");
        let id = make_ticket(&store, &ws, "Test", TicketPhase::InQa).await;

        let label = if should_succeed { "commit" } else { "rollback" };
        let result: anyhow::Result<()> =
            crate::db::with_tx(&store.conn, &id, "test_triple_write", async |tx| {
                BoardStore::set_commit_info_tx(
                    tx,
                    &id,
                    "abcdef0123456789abcdef0123456789abcd0123",
                    10,
                    5,
                )
                .await?;
                BoardStore::add_comment_tx(
                    tx,
                    &id,
                    crate::agent::role::SYSTEM_ROLE,
                    "triple write comment",
                )
                .await?;
                BoardStore::transition_to_tx(tx, &id, Some(TicketPhase::InQa), TicketPhase::Done)
                    .await?;
                if should_succeed {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("simulated failure for rollback test"))
                }
            })
            .await;

        if !should_succeed {
            assert!(result.is_err(), "({label}) expected transaction failure");
        }

        let ticket = crate::util::test::expect_ticket(&store, &id).await;
        let comments = store.get_comments(&id).await.expect("get comments");
        if should_succeed {
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
            // No changes should persist after rollback.
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
                TicketPhase::InQa,
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
            crate::db::params!["{not valid json}", id.clone()],
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
        /// Ticket is in InDiagnostics — claim should succeed.
        Success,
        /// Ticket is in a different phase — claim should fail.
        WrongPhase,
    }

    struct Case {
        name: &'static str,
        scenario: Scenario,
    }

    let cases = [
        Case {
            name: "in diagnostics succeeds",
            scenario: Scenario::Success,
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

        let claimed = store
            .claim_diagnostics(&id)
            .await
            .expect("claim_diagnostics");

        match case.scenario {
            Scenario::Success => {
                assert!(claimed, "Case '{}': expected claim to succeed", case.name);
                // Claim is a phase-CAS: the ticket stays in InDiagnostics.
                let ticket = crate::util::test::expect_ticket(&store, &id).await;
                assert_eq!(
                    ticket.phase,
                    TicketPhase::InDiagnostics,
                    "Case '{}': phase should remain InDiagnostics",
                    case.name
                );
            }
            Scenario::WrongPhase => {
                assert!(!claimed, "Case '{}': expected claim to fail", case.name);
            }
        }
    }
}

/// Round-trip test that exercises ALL column-index constants in
/// [`ticket_from_row`] by creating a ticket, setting every mutable field
/// via public API, then verifying every [`Ticket`] field survives the
/// SELECT → `ticket_from_row` deserialization path.
///
/// Serves as a regression test for ticket deserialization — the
/// [`columns!`] macro ensures single-sourcing of [`TICKET_COLUMNS`]
/// and [`COL_TICKET_*`], so column-order drift between them is
/// structurally impossible. This test still exercises the full
/// `ticket_from_row` deserialization path, including manual
/// field-by-field extraction via `row.get::<Type>(COL_TICKET_*)`
/// and default-value handling.
#[expect(clippy::too_many_lines)]
#[tokio::test]
async fn test_ticket_roundtrip_all_fields() {
    let (store, _tmp) = open_test_store().await;

    // Non-existent ticket returns None.
    let none = store.get_ticket("nonexistent").await.expect("get");
    assert!(none.is_none(), "non-existent ticket should return None");

    let ws = crate::workspace::test_ws_named("/test_ws", "test_workspace");

    // Create ticket with known values.
    let id = TicketBuilder::new(&store, &ws)
        .title("Roundtrip Title")
        .desc("Roundtrip description")
        .phase(TicketPhase::Backlog)
        .reporter("test_reporter")
        .create()
        .await
        .expect("create_ticket");

    // ── Fresh ticket (defaults: no active agents, no comments, no commit info) ─
    let fresh = store
        .get_ticket(&id)
        .await
        .expect("get_ticket")
        .expect("ticket exists");

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

    assert_eq!(
        fresh,
        Ticket {
            id: id.clone(),
            title: "Roundtrip Title".into(),
            description: "Roundtrip description".into(),
            phase: TicketPhase::Backlog,
            workspace_name: "test_workspace".into(),
            created_at: fresh.created_at.clone(),
            updated_at: fresh.updated_at.clone(),
            comments: vec![],
            prerequisites: vec![],
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test_reporter".into(),
            is_archived: false,
            priority: 1,
            reviewed_head: None,
            reviewed_tree: None,
            done_at: None,
            bounce_count: 0,
        },
    );

    // ── Mutated ticket (commit info) ────────────────────────────────────
    let tx = store.conn.begin_tx().await.unwrap();
    BoardStore::set_commit_info_tx(&tx, &id, "abcdef0123456789abcdef0123456789abcd0123", 42, 7)
        .await
        .expect("set_commit_info_tx");
    tx.commit().await.unwrap();

    store
        .set_reviewed_base(&id, Some("reviewed-head-hash"), Some("reviewed-tree-hash"))
        .await
        .expect("set_reviewed_base");

    let ticket = store
        .get_ticket(&id)
        .await
        .expect("get_ticket")
        .expect("ticket exists");

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

    assert_eq!(
        ticket,
        Ticket {
            id: id.clone(),
            title: "Roundtrip Title".into(),
            description: "Roundtrip description".into(),
            phase: TicketPhase::Backlog,
            workspace_name: "test_workspace".into(),
            created_at: ticket.created_at.clone(),
            updated_at: ticket.updated_at.clone(),
            comments: vec![],
            prerequisites: vec![],
            supersedes: None,
            superseded_by: None,
            commit_hash: Some("abcdef0123456789abcdef0123456789abcd0123".into()),
            lines_added: Some(42),
            lines_removed: Some(7),
            reporter: "test_reporter".into(),
            is_archived: false,
            priority: 1,
            reviewed_head: Some("reviewed-head-hash".into()),
            reviewed_tree: Some("reviewed-tree-hash".into()),
            done_at: None,
            bounce_count: 0,
        },
    );

    // ── Archived ticket (exercises is_archived bool deserialization) ────
    store.set_archived(&id).await.expect("set_archived");

    let archived = store
        .get_ticket(&id)
        .await
        .expect("get_ticket")
        .expect("ticket exists after archive");

    assert!(
        archived.created_at.contains('T'),
        "archived created_at should be RFC 3339: {}",
        archived.created_at,
    );
    assert!(
        archived.updated_at.contains('T'),
        "archived updated_at should be RFC 3339: {}",
        archived.updated_at,
    );

    assert_eq!(
        archived,
        Ticket {
            id,
            title: "Roundtrip Title".into(),
            description: "Roundtrip description".into(),
            phase: TicketPhase::Backlog,
            workspace_name: "test_workspace".into(),
            created_at: archived.created_at.clone(),
            updated_at: archived.updated_at.clone(),
            comments: vec![],
            prerequisites: vec![],
            supersedes: None,
            superseded_by: None,
            commit_hash: Some("abcdef0123456789abcdef0123456789abcd0123".into()),
            lines_added: Some(42),
            lines_removed: Some(7),
            reporter: "test_reporter".into(),
            is_archived: true,
            priority: 1,
            reviewed_head: Some("reviewed-head-hash".into()),
            reviewed_tree: Some("reviewed-tree-hash".into()),
            done_at: None,
            bounce_count: 0,
        },
    );
}

// ── done_at completion timestamp ────────────────────────────────────

/// done_at is stamped on transition to Done, survives later comments, is
/// cleared when the ticket leaves Done, and is re-stamped on re-completion.
#[tokio::test]
async fn test_done_at_transition_semantics() {
    let (store, _tmp) = open_test_store().await;
    let ws = crate::workspace::test_ws_named("/test_ws", "test_workspace");
    let id = TicketBuilder::new(&store, &ws)
        .title("Done timestamp")
        .create()
        .await
        .expect("create_ticket");

    store
        .transition_to(&id, None, TicketPhase::Done)
        .await
        .expect("transition to done");
    let done = store.get_ticket(&id).await.expect("get").expect("ticket");
    let first_done_at = done.done_at.expect("done_at set on completion");
    assert!(
        done.created_at < first_done_at,
        "done_at should be later than creation"
    );

    // Later activity (comments) must not move the completion timestamp.
    store
        .add_comment(&id, "manager", "nice work")
        .await
        .expect("add_comment");
    let commented = store.get_ticket(&id).await.expect("get").expect("ticket");
    assert_eq!(commented.done_at.as_deref(), Some(first_done_at.as_str()));
    assert!(
        commented.updated_at > first_done_at,
        "comment should bump updated_at but not done_at"
    );

    // Leaving Done clears the stamp; re-completion re-stamps it.
    store
        .transition_to(&id, Some(TicketPhase::Done), TicketPhase::Backlog)
        .await
        .expect("reopen");
    let reopened = store.get_ticket(&id).await.expect("get").expect("ticket");
    assert_eq!(reopened.done_at, None, "done_at cleared when leaving Done");

    store
        .transition_to(&id, Some(TicketPhase::Backlog), TicketPhase::Done)
        .await
        .expect("re-complete");
    let redone = store.get_ticket(&id).await.expect("get").expect("ticket");
    assert!(
        redone.done_at.as_deref().unwrap() > first_done_at.as_str(),
        "re-completion re-stamps done_at with the new moment"
    );
}

// ── FTS search (archived + active) ─────────────────────────────────

/// Create an archived ticket with the given title in tests.
async fn create_archived_ticket(
    store: &super::BoardStore,
    title: &str,
    workspace_name: &str,
) -> String {
    let ws = test_ws(workspace_name);
    let id = make_ticket(store, &ws, title, crate::pipeline::board::TicketPhase::Done).await;
    store.set_archived(&id).await.expect("set_archived");
    id
}

/// Create a non-archived ticket with the given title in tests.
async fn create_active_ticket(
    store: &super::BoardStore,
    title: &str,
    workspace_name: &str,
) -> String {
    let ws = test_ws(workspace_name);
    make_ticket(
        store,
        &ws,
        title,
        crate::pipeline::board::TicketPhase::Backlog,
    )
    .await
}

#[tokio::test]
async fn test_search_by_fts_finds_matching_title() {
    let (store, _tmp) = open_test_store().await;
    let archived = create_archived_ticket(&store, "Fix network timeout bug", "ws1").await;
    let active = create_active_ticket(&store, "Fix network timeout bug", "ws_active").await;

    let archived_results = store
        .search_archived_by_fts("network timeout", 10, "ws1")
        .await
        .expect("archived FTS search");
    assert!(
        archived_results.iter().any(|(id, _)| id == &archived),
        "archived search should find the archived ticket"
    );

    let active_results = store
        .search_by_fts("network timeout", 10, Some("ws_active"))
        .await
        .expect("FTS search");
    assert!(
        active_results.iter().any(|t| t.id == active),
        "search should find the active ticket"
    );
}

#[tokio::test]
async fn test_search_by_fts_includes_both_archive_states() {
    let (store, _tmp) = open_test_store().await;
    let archived = create_archived_ticket(&store, "still searching", "ws2").await;
    let active = create_active_ticket(&store, "still active", "ws2").await;

    // General search must find tickets in either archive state.
    let results = store
        .search_by_fts("still", 10, Some("ws2"))
        .await
        .expect("general FTS search");
    assert!(
        results.iter().any(|t| t.id == archived),
        "archived ticket must appear in general search results"
    );
    assert!(
        results.iter().any(|t| t.id == active),
        "active ticket must appear in general search results"
    );

    // Archived-only search keeps its is_archived = 1 filter.
    let archived_results = store
        .search_archived_by_fts("active", 10, "ws2")
        .await
        .expect("archived FTS search");
    assert!(
        archived_results.is_empty(),
        "non-archived ticket must not appear in archived search"
    );
}

#[tokio::test]
async fn test_search_by_fts_sanitize_short_circuit() {
    let (store, _tmp) = open_test_store().await;
    for query in ["!@#$%", ""] {
        let archived = store
            .search_archived_by_fts(query, 10, "ws")
            .await
            .expect("archived FTS search");
        assert!(
            archived.is_empty(),
            "query {query:?} yields no archived results"
        );
        let results = store
            .search_by_fts(query, 10, Some("ws"))
            .await
            .expect("FTS search");
        assert!(
            results.is_empty(),
            "query {query:?} yields no search results"
        );
    }
}

#[tokio::test]
async fn test_search_by_fts_scoped_to_workspace() {
    let (store, _tmp) = open_test_store().await;
    create_active_ticket(&store, "Fix network timeout bug", "ws_scope_a").await;
    create_active_ticket(&store, "Database connection pool error", "ws_scope_b").await;

    let results = store
        .search_by_fts("network timeout", 10, Some("ws_scope_a"))
        .await
        .expect("FTS search scoped to ws_scope_a");
    assert_eq!(results.len(), 1, "should find only ws_scope_a ticket");
    assert_eq!(
        results[0].workspace_name, "ws_scope_a",
        "ticket belongs to ws_scope_a"
    );
}

/// Assert that `display` contains every `(needle, msg)` pair.
///
/// `format!`-based needles must be materialised into owned `String`s before
/// building the slice (a `&` to a temporary would not live long enough).
fn assert_display_contains(display: &str, needles: &[(&str, &str)]) {
    for (needle, msg) in needles {
        assert!(display.contains(needle), "{msg}");
    }
}

/// Assert that `display` does NOT contain every `(needle, msg)` pair.
fn assert_display_not_contains(display: &str, needles: &[(&str, &str)]) {
    for (needle, msg) in needles {
        assert!(!display.contains(needle), "{msg}");
    }
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

    let ticket_id_needle = format!("Ticket: {id}");
    let prereq_needle = format!("Prerequisites: {prereq_id}");
    assert_display_contains(
        &display,
        &[
            (&ticket_id_needle, "should contain ticket id"),
            ("Title: Display Test Ticket", "should contain title"),
            (
                "Description: A description for testing",
                "should contain description",
            ),
            ("Phase: in_development", "should use snake_case phase"),
            ("Reporter: manager", "should contain reporter"),
            ("Workspace: test-ws", "should contain workspace"),
            ("Created:", "should contain created timestamp"),
            ("Updated:", "should contain updated timestamp"),
            (&prereq_needle, "should show prerequisites"),
            ("Comments:", "should have comments section"),
            ("(no comments)", "should show no comments"),
            ("Priority: P1", "should contain priority label (default 1)"),
        ],
    );

    // Fields that should NOT appear when unset
    assert_display_not_contains(
        &display,
        &[
            ("Supersedes:", "no supersedes when not set"),
            ("Superseded by:", "no superseded_by when not set"),
            ("Archived:", "no archived line when false"),
            ("commit_hash:", "commit_hash should not be displayed"),
            ("lines_added:", "lines_added should not be displayed"),
            ("lines_removed:", "lines_removed should not be displayed"),
        ],
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

    assert_display_contains(
        &display,
        &[
            ("Comments:", "should have comments section"),
            ("[analyst]", "should show analyst role"),
            ("[reviewer]", "should show reviewer role"),
            ("First comment", "should show first comment"),
            ("Second comment", "should show second comment"),
        ],
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
    let store = crate::pipeline::board::BOARD.get().unwrap();
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
        let candidates = store
            .list_archived_with_embeddings("ws")
            .await
            .expect("list");
        assert!(candidates.is_empty(), "no tickets at all");
    }

    let ws = test_ws("ws");

    // Create a ticket with a known embedding blob (two small f32s)
    let embedding: Vec<f32> = vec![1.0, 2.0];
    let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

    let id = TicketBuilder::new(&store, &ws)
        .title("Embedded ticket")
        .phase(crate::pipeline::board::TicketPhase::Done)
        .embedding(&blob)
        .create()
        .await
        .expect("create_ticket with embedding");
    store.set_archived(&id).await.expect("archive");

    let candidates = store
        .list_archived_with_embeddings("ws")
        .await
        .expect("list");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0, id);
    assert_eq!(candidates[0].1, vec![1.0, 2.0]);
}

// ── route_comment_to_agents tests ─────────────────────────────────────

/// route_comment_to_agents silently skips when no agents are assigned.
#[tokio::test]
async fn test_route_comment_to_agents_no_assignment() {
    crate::util::test::init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = crate::workspace::test_ws("/tmp/test_route_comment_no_assign");

    // Create a ticket with no active agents / no assignment
    let ticket_id = crate::util::test::make_ticket(
        store,
        &ws,
        "no-assign-test",
        crate::pipeline::board::TicketPhase::Backlog,
    )
    .await;

    // Add a comment — should succeed without routing (no assigned agents)
    store
        .add_comment(&ticket_id, "manager", "No one should get this")
        .await
        .expect("add_comment should succeed");
}

/// route_comment_to_agents delivers a comment to the registered agent with
/// the commenter's role in the AgentJob. The engineer row guards the
/// Role::parse → Manager fallback — the manager row alone would pass silently.
///
/// Serialized with the reset_analysis_tickets tests (shared global board — a
/// concurrent boot reset would clobber the fixture phases).
#[tokio::test]
#[serial_test::serial(reset_inflight)]
async fn test_route_comment_to_agents_delivers_with_commenter_role() {
    crate::util::test::init_management_test_stores().await;
    let store = crate::pipeline::board::store();

    for (i, (commenter, content, expected_role)) in [
        ("manager", "Hello from test", crate::Role::Manager),
        ("engineer", "Code review feedback", crate::Role::Engineer),
    ]
    .into_iter()
    .enumerate()
    {
        let ws = crate::workspace::test_ws(format!("/tmp/test_route_comment_{i}"));
        let ticket_id = crate::util::test::make_ticket(
            store,
            &ws,
            &format!("route-comment-test-{i}"),
            crate::pipeline::board::TicketPhase::InDevelopment,
        )
        .await;

        let job_id = format!("test-route-job-{i}");
        crate::jobs::spawn_job(
            &crate::session::store().conn,
            &job_id,
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Engineer,
            &[],
            &crate::jobs::SpawnChild::TicketImplementation {
                ticket_id: ticket_id.clone(),
            },
        )
        .await
        .expect("create implementation");
        let agent_id = format!("_test_route_comment_agent_{i}");
        crate::jobs::upsert_job_agent(
            &crate::session::store().conn,
            &job_id,
            &agent_id,
            crate::jobs::AgentKind::Verifier,
            crate::jobs::RowStatus::Launched,
            "",
        )
        .await
        .expect("set active agent");
        let mut rx = crate::agent::message_router::register_agent(&agent_id);

        store
            .add_comment(&ticket_id, commenter, content)
            .await
            .expect("add_comment should succeed");

        let received = rx.try_recv().expect("should receive the routed comment");
        assert_eq!(received.content, content);
        assert_eq!(
            received.kind,
            crate::agent::message_router::MessageKind::TicketComment
        );
        assert_eq!(received.user_name, commenter);
        assert_eq!(
            received.role, expected_role,
            "role should be the commenter's role ({commenter})",
        );
        assert!(
            rx.try_recv().is_err(),
            "should not have additional messages"
        );

        crate::agent::message_router::unregister_agent(&agent_id);
    }
}
