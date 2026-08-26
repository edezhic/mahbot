//! Core pipeline flow tests — the job-per-phase single-puller orchestration,
//! claims, bounce/reset failure semantics, and the phase machine.
//!
//! These exercise the deterministic store + orchestrator paths without invoking
//! agent LLM rounds. Global-DB tests are serialized behind [`TEST_LOCK`] because
//! every global store shares one test root; isolated-store tests run freely.

use crate::pipeline::board::{BoardStore, PipelineCheck, TicketPhase};
use crate::util::test::{
    create_test_workspace, expect_ticket, init_management_test_stores, make_ticket,
};
use crate::{Workspace, WorkspaceStatus};

/// Serializes tests that mutate the global stores (they share the test DB).
/// Async-aware so the guard may be held across `.await` points safely.
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn ws_named(name: &str) -> Workspace {
    crate::workspace::test_ws_named("/tmp/ws", name)
}

/// Claim Backlog → Analysis must wait out the claim grace; a ticket aged past
/// it is claimed atomically.
#[tokio::test]
async fn claim_backlog_obeys_grace() {
    let (store, _dir) = crate::open_test_store!(BoardStore, "board");
    let ws = ws_named("ws-a");
    let id = make_ticket(&store, &ws, "Grace ticket", TicketPhase::Backlog).await;

    // Fresh ticket sits inside the grace window → no claim.
    assert!(
        store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::Analysis,
                &ws.name,
                PipelineCheck::Skip,
                Some(BoardStore::BACKLOG_CLAIM_GRACE),
            )
            .await
            .unwrap()
            .is_none(),
        "fresh ticket must not be claimed inside the grace window",
    );

    // Age the ticket past the grace window → the claim succeeds.
    let past = (chrono::Utc::now() - chrono::Duration::seconds(10)).to_rfc3339();
    store
        .conn
        .execute(
            "UPDATE tickets SET created_at = ?1 WHERE id = ?2",
            crate::db::params![past, id.clone()],
        )
        .await
        .unwrap();
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            &ws.name,
            PipelineCheck::Skip,
            Some(BoardStore::BACKLOG_CLAIM_GRACE),
        )
        .await
        .unwrap()
        .expect("aged ticket must be claimed");
    assert_eq!(claimed.id, id);
    assert_eq!(claimed.phase, TicketPhase::Analysis);
}

/// ReadyForDevelopment → InDevelopment is gated on single pipeline occupancy:
/// a candidate is blocked exactly while a sibling sits in a pipeline phase.
#[tokio::test]
async fn claim_rfd_blocks_on_pipeline_occupancy() {
    let (store, _dir) = crate::open_test_store!(BoardStore, "board");
    let ws = ws_named("ws-a");

    // No occupied sibling yet → the RFD claim succeeds.
    let first = make_ticket(&store, &ws, "RFD one", TicketPhase::ReadyForDevelopment).await;
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::ReadyForDevelopment,
            TicketPhase::InDevelopment,
            &ws.name,
            PipelineCheck::Enforce,
            None,
        )
        .await
        .unwrap()
        .expect("RFD ticket must claim without a pipeline-occupied sibling");
    assert_eq!(claimed.id, first);

    // `first` is now InDevelopment (a pipeline-occupied phase) → a second RFD
    // ticket is blocked.
    make_ticket(&store, &ws, "RFD two", TicketPhase::ReadyForDevelopment).await;
    assert!(
        store
            .claim_ticket_in_workspace(
                TicketPhase::ReadyForDevelopment,
                TicketPhase::InDevelopment,
                &ws.name,
                PipelineCheck::Enforce,
                None,
            )
            .await
            .unwrap()
            .is_none(),
        "RFD claim must be blocked while a pipeline-occupied sibling exists",
    );
}

/// Backlog → Analysis claim honors prerequisites: a dependent ticket with an
/// unmet prereq stays in Backlog; once the prereq reaches an unblocking phase
/// the dependent is eligible.
#[tokio::test]
async fn claim_backlog_honors_prerequisites() {
    let (store, _dir) = crate::open_test_store!(BoardStore, "board");
    let ws = ws_named("ws-a");

    // The prereq lives OUTSIDE Backlog so the only Backlog candidate is the
    // dependent ticket (a Backlog prereq would be claimed itself).
    let prereq_analysis =
        make_ticket(&store, &ws, "Prereq (analysis)", TicketPhase::Analysis).await;
    let dependent = crate::util::test::TicketBuilder::new(&store, &ws)
        .title("Dependent")
        .phase(TicketPhase::Backlog)
        .prereqs(std::slice::from_ref(&prereq_analysis))
        .create()
        .await
        .unwrap();

    // Unmet prereq (its prereq is not in UNBLOCKING_PHASES) → blocked.
    assert!(
        store
            .claim_ticket_in_workspace(
                TicketPhase::Backlog,
                TicketPhase::Analysis,
                &ws.name,
                PipelineCheck::Skip,
                None,
            )
            .await
            .unwrap()
            .is_none(),
        "dependent with a non-unblocking prereq must not be claimed",
    );

    // Move the prereq to Done (an unblocking phase) → the dependent claims.
    store
        .transition_to(
            &prereq_analysis,
            Some(TicketPhase::Analysis),
            TicketPhase::Done,
        )
        .await
        .unwrap();
    let claimed = store
        .claim_ticket_in_workspace(
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            &ws.name,
            PipelineCheck::Skip,
            None,
        )
        .await
        .unwrap()
        .expect("dependent must claim once its prereq unblocks");
    assert_eq!(claimed.id, dependent);
}

/// A phase job is unique per (kind, ticket_id); creating a duplicate is a
/// no-op error, and completing the job removes it. Runs on the consolidated
/// test DB because `jobs.ticket_id` references the `tickets` table.
#[tokio::test]
async fn phase_job_is_unique_and_terminalizes() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/jobs_ws", "jobs_ws").await;
    let ticket_id = make_ticket(store, &ws, "Jobs", TicketPhase::InDevelopment).await;
    let conn = &crate::session::store().conn;
    let phase = TicketPhase::InDevelopment;
    let job_id = crate::generate_id();

    crate::jobs::spawn_job(
        conn,
        &job_id,
        "task",
        &ws.name,
        "",
        "",
        crate::Role::Engineer,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase,
            ticket_id: ticket_id.clone(),
        },
    )
    .await
    .unwrap();

    // The (kind, ticket_id) index makes a second job for the same phase a no-op.
    crate::jobs::spawn_job(
        conn,
        &crate::generate_id(),
        "task",
        &ws.name,
        "",
        "",
        crate::Role::Engineer,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase,
            ticket_id: ticket_id.clone(),
        },
    )
    .await
    .expect_err("duplicate phase job must be rejected by the unique index");

    let row = crate::jobs::find_phase_job(conn, &ticket_id, phase)
        .await
        .unwrap()
        .expect("phase job must be found");
    assert_eq!(row.id, job_id);

    crate::jobs::complete_ticket_job(conn, &job_id)
        .await
        .unwrap();
    assert!(
        crate::jobs::find_phase_job(conn, &ticket_id, phase)
            .await
            .unwrap()
            .is_none(),
        "completed phase job must be removed",
    );
}

/// The phase machine: pipeline-occupied, terminal, and unblocking sets are
/// mutually consistent.
#[tokio::test]
async fn phase_machine_classifications_are_consistent() {
    use strum::IntoEnumIterator;
    for phase in TicketPhase::iter() {
        let occupied = phase.is_pipeline_occupied();
        let terminal = phase.is_terminal();
        assert!(
            !(occupied && terminal),
            "{phase} both occupied and terminal",
        );
        if occupied {
            assert!(!phase.is_unblocking(), "{phase} occupied but unblocking");
        }
    }
}

/// The puller claim driver claims Backlog → Analysis and RFD → InDevelopment
/// for a live (Ready) workspace.
#[tokio::test]
async fn run_claim_pipeline_claims_new_work() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/launch_ws", "launch_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();

    let backlog_id = make_ticket(store, &ws, "Backlog", TicketPhase::Backlog).await;
    let rfd_id = make_ticket(store, &ws, "RFD", TicketPhase::ReadyForDevelopment).await;

    super::run_claim_pipeline(&ws).await;

    // Backlog ticket is too fresh (grace) → stays Backlog.
    assert_eq!(
        crate::util::test::expect_ticket_phase(store, &backlog_id).await,
        TicketPhase::Backlog,
    );
    // RFD has no grace → claims immediately.
    assert_eq!(
        crate::util::test::expect_ticket_phase(store, &rfd_id).await,
        TicketPhase::InDevelopment,
    );
}

/// A non-exhausting bounce moves the ticket back to InDevelopment and
/// increments the bounce counter (no breaker trip, no workspace pause).
#[tokio::test]
async fn bounce_to_development_returns_ticket_without_tripping() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/bounce_ws", "bounce_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Bounce", TicketPhase::InReview).await;
    let job_id = crate::generate_id();
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        &job_id,
        "review",
        &ws.name,
        "",
        "",
        crate::Role::Reviewer,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase: TicketPhase::InReview,
            ticket_id: id.clone(),
        },
    )
    .await
    .unwrap();

    super::bounce_to_development(
        &expect_ticket(store, &id).await,
        TicketPhase::InReview,
        "Reviewers",
        true,
        "Reviewer",
        "failed",
        &job_id,
        &ws,
    )
    .await;

    let ticket = expect_ticket(store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::InDevelopment);
    assert_eq!(ticket.bounce_count, 1);
    assert!(
        crate::jobs::find_phase_job(&crate::session::store().conn, &id, TicketPhase::InReview)
            .await
            .unwrap()
            .is_none(),
        "a non-trip bounce deletes the phase job so the puller re-dispatches",
    );
}

/// Exhausting the bounce budget trips the breaker: the ticket → Failed, the
/// phase job is deleted, and no bounce count is consumed further.
#[tokio::test]
async fn bounce_breaker_trips_to_failed() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/trip_ws", "trip_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Trip", TicketPhase::InReview).await;
    let job_id = crate::generate_id();
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        &job_id,
        "review",
        &ws.name,
        "",
        "",
        crate::Role::Reviewer,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase: TicketPhase::InReview,
            ticket_id: id.clone(),
        },
    )
    .await
    .unwrap();
    // Seed the bounce budget at the exhaustion threshold.
    let budget: i64 = 10;
    store
        .conn
        .execute(
            "UPDATE tickets SET bounce_count = ?1 WHERE id = ?2",
            crate::db::params![budget, id.clone()],
        )
        .await
        .unwrap();

    super::bounce_to_development(
        &expect_ticket(store, &id).await,
        TicketPhase::InReview,
        "Reviewers",
        true,
        "Reviewer",
        "failed",
        &job_id,
        &ws,
    )
    .await;

    let ticket = expect_ticket(store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::Failed);
    assert_eq!(ticket.bounce_count, budget); // terminal: no further increment
}

/// `reset_phase_attempt` destroys the current attempt: it comments, cancels
/// orphaned agents, pauses the workspace for an implementation phase, and
/// deletes the phase job so the puller creates a fresh one.
#[tokio::test]
async fn reset_phase_attempt_destroys_attempt_and_pauses() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/reset_ws", "reset_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Reset", TicketPhase::InDevelopment).await;
    let job_id = crate::generate_id();
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        &job_id,
        "dev",
        &ws.name,
        "",
        "",
        crate::Role::Engineer,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase: TicketPhase::InDevelopment,
            ticket_id: id.clone(),
        },
    )
    .await
    .unwrap();

    super::reset_phase_attempt(
        &expect_ticket(store, &id).await,
        TicketPhase::InDevelopment,
        &job_id,
        "test hard failure",
        "attempt reset",
    )
    .await;

    assert!(
        crate::jobs::find_phase_job(
            &crate::session::store().conn,
            &id,
            TicketPhase::InDevelopment
        )
        .await
        .unwrap()
        .is_none(),
        "reset must delete the phase job",
    );
    // Implementation-phase hard failure pauses the workspace.
    let ws_row = crate::workspace::store()
        .get_by_name(&ws.name)
        .await
        .unwrap()
        .unwrap();
    assert!(
        ws_row.paused,
        "implementation-phase reset must auto-pause the workspace",
    );
    let comments = store.get_comments(&id).await.unwrap();
    assert!(
        comments.iter().any(|c| c.content.contains("attempt reset")),
        "reset must leave an explanatory comment",
    );
}

/// Analysis is not an implementation phase: a hard-failure reset there must NOT
/// pause the workspace, while still deleting the phase job.
#[tokio::test]
async fn reset_phase_attempt_analysis_does_not_pause() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/analysis_ws", "analysis_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Analysis", TicketPhase::Analysis).await;
    let job_id = crate::generate_id();
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        &job_id,
        "analysis",
        &ws.name,
        "",
        "",
        crate::Role::Analyst,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase: TicketPhase::Analysis,
            ticket_id: id.clone(),
        },
    )
    .await
    .unwrap();

    super::reset_phase_attempt(
        &expect_ticket(store, &id).await,
        TicketPhase::Analysis,
        &job_id,
        "analysis failure",
        "analysis reset",
    )
    .await;

    assert!(
        crate::jobs::find_phase_job(&crate::session::store().conn, &id, TicketPhase::Analysis)
            .await
            .unwrap()
            .is_none(),
    );
    let ws_row = crate::workspace::store()
        .get_by_name(&ws.name)
        .await
        .unwrap()
        .unwrap();
    assert!(
        !ws_row.paused,
        "analysis reset must NOT pause the workspace",
    );
}
