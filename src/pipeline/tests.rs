//! Core pipeline flow tests — the job-per-phase single-puller orchestration,
//! claims, bounce/reset failure semantics, and the phase machine.
//!
//! The deterministic store + orchestrator tests exercise claim/bounce/reset and
//! phase classification without agent rounds. The end-to-end behavioral-oracle
//! tests drive the real phase bodies (analysis / development / diagnostics /
//! review / QA / sanitation) and the puller through a scripted
//! [`FakeProvider`](crate::util::test::FakeProvider), so they are isolated from
//! the live model. Global-DB tests are serialized behind [`TEST_LOCK`] because
//! every global store shares one test root; isolated-store tests run freely.

use crate::pipeline::board::{BoardStore, Ticket, TicketPhase};
use crate::util::test::{
    create_test_workspace, expect_ticket, expect_ticket_phase, init_management_test_stores,
    make_ticket,
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
    let (now, cutoff) = claim_clock();
    assert!(
        store
            .claim_backlog_for_analysis(&ws.name, now, cutoff)
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
    let (now, cutoff) = claim_clock();
    let claimed = store
        .claim_backlog_for_analysis(&ws.name, now, cutoff)
        .await
        .unwrap()
        .expect("aged ticket must be claimed");
    assert_eq!(claimed.id, id);
    assert_eq!(claimed.phase, TicketPhase::Analysis);
}

/// Queued → InDevelopment is gated on single pipeline occupancy:
/// a candidate is blocked exactly while a sibling sits in a pipeline phase.
#[tokio::test]
async fn claim_queued_blocks_on_pipeline_occupancy() {
    let (store, _dir) = crate::open_test_store!(BoardStore, "board");
    let ws = ws_named("ws-a");

    // No occupied sibling yet → the Queued claim succeeds.
    let first = make_ticket(&store, &ws, "Queued one", TicketPhase::Queued).await;
    let claimed = store
        .claim_queued_for_development(&ws.name, crate::db::now())
        .await
        .unwrap()
        .expect("Queued ticket must claim without a pipeline-occupied sibling");
    assert_eq!(claimed.id, first);

    // `first` is now InDevelopment (a pipeline-occupied phase) → a second Queued
    // ticket is blocked.
    make_ticket(&store, &ws, "Queued two", TicketPhase::Queued).await;
    assert!(
        store
            .claim_queued_for_development(&ws.name, crate::db::now())
            .await
            .unwrap()
            .is_none(),
        "Queued claim must be blocked while a pipeline-occupied sibling exists",
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

    // Unmet prereq (its prereq is not in UNBLOCKING_PHASES) → blocked. Age the
    // dependent past the grace window FIRST so this assertion isolates the
    // prereq rule rather than passing vacuously on the grace cutoff.
    age_ticket_past_grace(&store, &dependent).await;
    let (now, cutoff) = claim_clock();
    assert!(
        store
            .claim_backlog_for_analysis(&ws.name, now, cutoff)
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
    let (now, cutoff) = claim_clock();
    let claimed = store
        .claim_backlog_for_analysis(&ws.name, now, cutoff)
        .await
        .unwrap()
        .expect("dependent must claim once its prereq unblocks");
    assert_eq!(claimed.id, dependent);
}

/// Queued → InDevelopment honors prerequisites too (the same filter the
/// Backlog claim applies): a Queued ticket with an unmet prereq stays queued
/// even with a free pipeline, and claims once the prereq unblocks.
#[tokio::test]
async fn claim_queued_honors_prerequisites() {
    let (store, _dir) = crate::open_test_store!(BoardStore, "board");
    let ws = ws_named("ws-a");

    // The prereq lives OUTSIDE Queued so the only Queued candidate is the
    // dependent ticket (a Queued prereq would be claimed itself).
    let prereq_analysis =
        make_ticket(&store, &ws, "Prereq (analysis)", TicketPhase::Analysis).await;
    let dependent = crate::util::test::TicketBuilder::new(&store, &ws)
        .title("Dependent")
        .phase(TicketPhase::Queued)
        .prereqs(std::slice::from_ref(&prereq_analysis))
        .create()
        .await
        .unwrap();

    // Unmet prereq → blocked despite a free pipeline.
    assert!(
        store
            .claim_queued_for_development(&ws.name, crate::db::now())
            .await
            .unwrap()
            .is_none(),
        "Queued ticket with a non-unblocking prereq must not be claimed",
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
        .claim_queued_for_development(&ws.name, crate::db::now())
        .await
        .unwrap()
        .expect("dependent must claim once its prereq unblocks");
    assert_eq!(claimed.id, dependent);
    assert_eq!(claimed.phase, TicketPhase::InDevelopment);
}

/// The claim probes are a fail-open cost gate, so a false negative (probe
/// says "no candidate" while the claim would succeed) is the only forbidden
/// outcome. Differential oracle: with a shared clock the probe decision must
/// equal the claim outcome in every eligibility scenario — grace boundary,
/// prerequisites, and pipeline occupancy.
#[tokio::test]
async fn claim_probes_agree_with_claim_outcomes() {
    // ── Backlog grace boundary: fresh inside grace vs aged past it ───────
    {
        let (store, _dir) = crate::open_test_store!(BoardStore, "board");
        let ws = ws_named("probe-blog-grace");
        let id = make_ticket(&store, &ws, "Fresh backlog", TicketPhase::Backlog).await;

        let (now, cutoff) = claim_clock();
        let (probe, claimed) = probe_then_claim_backlog(&store, &ws, now, &cutoff).await;
        assert!(!probe, "fresh Backlog inside grace must probe false");
        assert!(
            claimed.is_none(),
            "fresh Backlog inside grace must not claim",
        );

        age_ticket_past_grace(&store, &id).await;
        let (now, cutoff) = claim_clock();
        let (probe, claimed) = probe_then_claim_backlog(&store, &ws, now, &cutoff).await;
        assert!(probe, "aged Backlog past grace must probe true");
        assert!(claimed.is_some(), "aged Backlog past grace must claim");
    }

    // ── Backlog prerequisites: unmet blocks, unblocking frees ───────────
    {
        let (store, _dir) = crate::open_test_store!(BoardStore, "board");
        let ws = ws_named("probe-blog-prereq");
        let prereq_analysis =
            make_ticket(&store, &ws, "Prereq (analysis)", TicketPhase::Analysis).await;
        let dependent = crate::util::test::TicketBuilder::new(&store, &ws)
            .title("Dependent")
            .phase(TicketPhase::Backlog)
            .prereqs(std::slice::from_ref(&prereq_analysis))
            .create()
            .await
            .unwrap();
        age_ticket_past_grace(&store, &dependent).await;

        let (now, cutoff) = claim_clock();
        let (probe, claimed) = probe_then_claim_backlog(&store, &ws, now, &cutoff).await;
        assert!(
            !probe,
            "dependent with a non-unblocking prereq must probe false",
        );
        assert!(
            claimed.is_none(),
            "dependent with an unmet prereq must not claim",
        );

        store
            .transition_to(
                &prereq_analysis,
                Some(TicketPhase::Analysis),
                TicketPhase::Done,
            )
            .await
            .unwrap();
        let (now, cutoff) = claim_clock();
        let (probe, claimed) = probe_then_claim_backlog(&store, &ws, now, &cutoff).await;
        assert!(probe, "dependent must probe true once its prereq unblocks");
        assert!(
            claimed.is_some(),
            "dependent must claim once its prereq unblocks"
        );
    }

    // ── Queued: unmet prereq (free pipeline) blocks; unblocking claims; a
    //    sibling then occupies the pipeline and blocks a second candidate.
    {
        let (store, _dir) = crate::open_test_store!(BoardStore, "board");
        let ws = ws_named("probe-queued");
        let prereq_analysis =
            make_ticket(&store, &ws, "Prereq (analysis)", TicketPhase::Analysis).await;
        let dependent = crate::util::test::TicketBuilder::new(&store, &ws)
            .title("Dependent")
            .phase(TicketPhase::Queued)
            .prereqs(std::slice::from_ref(&prereq_analysis))
            .create()
            .await
            .unwrap();

        // Unmet prereq with a free pipeline → blocked.
        let (probe, claimed) = probe_then_claim_queued(&store, &ws).await;
        assert!(
            !probe,
            "Queued with a non-unblocking prereq must probe false",
        );
        assert!(
            claimed.is_none(),
            "Queued with an unmet prereq must not claim",
        );

        // Prereq unblocks → free pipeline → claims (now InDevelopment → occupied).
        store
            .transition_to(
                &prereq_analysis,
                Some(TicketPhase::Analysis),
                TicketPhase::Done,
            )
            .await
            .unwrap();
        let (probe, claimed) = probe_then_claim_queued(&store, &ws).await;
        assert!(probe, "Queued with a free pipeline must probe true");
        let claimed = claimed.expect("dependent must claim once its prereq unblocks");
        assert_eq!(claimed.phase, TicketPhase::InDevelopment);
        assert_eq!(claimed.id, dependent);

        // The claimed ticket now occupies the pipeline → blocks the next Queued
        // candidate.
        make_ticket(&store, &ws, "Queued two", TicketPhase::Queued).await;
        let (probe, claimed) = probe_then_claim_queued(&store, &ws).await;
        assert!(!probe, "Queued with an occupied sibling must probe false");
        assert!(
            claimed.is_none(),
            "Queued with an occupied sibling must not claim",
        );
    }
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
        None,
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
        None,
    )
    .await
    .expect_err("duplicate phase job must be rejected by the unique index");

    let row = crate::jobs::find_phase_job(conn, &ticket_id, phase)
        .await
        .unwrap()
        .expect("phase job must be found");
    assert_eq!(row.id, job_id);

    crate::jobs::terminalize_job(conn, &job_id).await.unwrap();
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

/// The puller claim driver claims Backlog → Analysis and Queued → InDevelopment
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
    let queued_id = make_ticket(store, &ws, "Queued", TicketPhase::Queued).await;

    super::run_claim_pipeline(&ws).await;

    // Backlog ticket is too fresh (grace) → stays Backlog.
    assert_eq!(
        crate::util::test::expect_ticket_phase(store, &backlog_id).await,
        TicketPhase::Backlog,
    );
    // Queued has no grace → claims immediately.
    assert_eq!(
        crate::util::test::expect_ticket_phase(store, &queued_id).await,
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
        None,
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
        None,
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
        None,
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
        None,
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

// ── End-to-end behavioral-oracle helpers ────────────────────────────────

/// Create a phase job row without spawning a body — the test drives the phase
/// body directly so the provider script is consumed deterministically.
async fn spawn_phase_job(
    job_id: &str,
    ws: &Workspace,
    ticket_id: &str,
    phase: TicketPhase,
    role: crate::Role,
    task: &str,
) {
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        job_id,
        task,
        &ws.name,
        "",
        "",
        role,
        &[],
        &crate::jobs::SpawnChild::Phase {
            phase,
            ticket_id: ticket_id.to_string(),
        },
        None,
    )
    .await
    .unwrap();
}

/// Find the launched phase job for a ticket, if present.
async fn expect_phase_job(
    store: &BoardStore,
    id: &str,
    phase: TicketPhase,
) -> Option<crate::jobs::TicketJobRow> {
    crate::jobs::find_phase_job(&store.conn, id, phase)
        .await
        .unwrap()
}

/// A scripted FakeProvider that yields `count` clean verifier pairs: each
/// agent produces a turn response then a clean `{"score":10,"issues":[]}`
/// verdict.
fn fake_clean_verifiers(count: usize) -> crate::util::test::FakeProvider {
    let mut fake = crate::util::test::FakeProvider::new();
    for _ in 0..count {
        fake = fake
            .ok("verifier check ok")
            .ok(r#"{"score":10,"issues":[]}"#);
    }
    fake
}

/// Poll (tightly) until an agent for `ticket_id` is registered in the global
/// registry, or panic after `timeout`.
async fn wait_for_agent_registered(ticket_id: &str, timeout: std::time::Duration) {
    tokio::time::timeout(timeout, async {
        loop {
            if crate::agent::registry::AGENT_REGISTRY.has_agents_for_ticket(ticket_id) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("agent must register within the timeout");
}

/// Assert the workspace row's `paused` column equals `expected`.
async fn assert_workspace_paused(ws: &Workspace, expected: bool) {
    let ws_row = crate::workspace::store()
        .get_by_name(&ws.name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        ws_row.paused, expected,
        "workspace {} paused={} expected {expected}",
        ws.name, ws_row.paused,
    );
}

/// Age a ticket's `created_at` past the Backlog claim grace so the puller will
/// claim it.
async fn age_ticket_past_grace(store: &BoardStore, id: &str) {
    let past = (chrono::Utc::now() - chrono::Duration::seconds(10)).to_rfc3339();
    store
        .conn
        .execute(
            "UPDATE tickets SET created_at = ?1 WHERE id = ?2",
            crate::db::params![past, id.to_string()],
        )
        .await
        .unwrap();
}

/// Claim-time clock shared by claim calls in tests: now + the grace cutoff
/// derived from it (same instant, as in production).
fn claim_clock() -> (String, String) {
    let now = crate::db::now();
    let cutoff = (chrono::Utc::now() - BoardStore::BACKLOG_CLAIM_GRACE).to_rfc3339();
    (now, cutoff)
}

/// Probe then claim Backlog→Analysis under a shared clock, asserting the
/// probe decision equals the claim outcome (the differential invariant).
async fn probe_then_claim_backlog(
    store: &BoardStore,
    ws: &Workspace,
    now: String,
    cutoff: &str,
) -> (bool, Option<Ticket>) {
    let probe = store
        .backlog_claim_candidate_exists(&ws.name, cutoff)
        .await
        .unwrap();
    let claimed = store
        .claim_backlog_for_analysis(&ws.name, now, cutoff.to_string())
        .await
        .unwrap();
    assert_eq!(
        probe,
        claimed.is_some(),
        "probe must agree with claim outcome",
    );
    (probe, claimed)
}

/// Probe then claim Queued→InDevelopment, asserting the probe decision equals
/// the claim outcome (the differential invariant).
async fn probe_then_claim_queued(store: &BoardStore, ws: &Workspace) -> (bool, Option<Ticket>) {
    let probe = store.queued_claim_candidate_exists(&ws.name).await.unwrap();
    let claimed = store
        .claim_queued_for_development(&ws.name, crate::db::now())
        .await
        .unwrap();
    assert_eq!(
        probe,
        claimed.is_some(),
        "probe must agree with claim outcome",
    );
    (probe, claimed)
}

// ── 1. Full pipeline lifecycle ──────────────────────────────────────────

/// Drive a Backlog→Done lifecycle through the real phase bodies and the
/// per-phase puller claims: analysis → planning → queued → development →
/// diagnostics (skipped) → review (skip-reviewed) → QA → sanitation (dirty
/// commit).

#[serial_test::serial(provider)]
#[tokio::test]
async fn full_pipeline_lifecycle_backlog_to_done_with_skip_review_and_dirty_commit() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let (repo_dir, repo_path) = crate::util::test::init_temp_repo();
    let ws = create_test_workspace(repo_path.to_str().unwrap(), "lifecycle_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Lifecycle", TicketPhase::Backlog).await;
    age_ticket_past_grace(store, &id).await;

    // Puller claim Backlog → Analysis.
    super::run_claim_pipeline(&ws).await;
    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::Analysis,
        "aged backlog ticket must be claimed into Analysis",
    );

    // Analysis: 3 clean analysts → Planning, job deleted.
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::Analysis,
        crate::Role::Analyst,
        "analysis",
    )
    .await;
    {
        let fake = crate::util::test::FakeProvider::new()
            .ok("analyst one")
            .ok(r#"{"issues":[]}"#)
            .ok("analyst two")
            .ok(r#"{"issues":[]}"#)
            .ok("analyst three")
            .ok(r#"{"issues":[]}"#);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::analysis::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }
    assert_eq!(expect_ticket_phase(store, &id).await, TicketPhase::Planning);
    assert!(
        expect_phase_job(store, &id, TicketPhase::Analysis)
            .await
            .is_none(),
        "a completed analysis round deletes the phase job",
    );

    // Manual Planning → Queued, then puller claim Queued → InDevelopment.
    store
        .transition_to(&id, Some(TicketPhase::Planning), TicketPhase::Queued)
        .await
        .unwrap();
    super::run_claim_pipeline(&ws).await;
    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::InDevelopment,
    );

    // Engineer: 2 responses → InDevelopment → InDiagnostics.
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InDevelopment,
        crate::Role::Engineer,
        "implement",
    )
    .await;
    {
        let fake = crate::util::test::FakeProvider::new()
            .ok("implemented the ticket")
            .ok(r#"{"items":["did the thing"],"summary":"done"}"#);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::development::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }
    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::InDiagnostics,
    );

    // Diagnostics: no commands configured → skipped, no provider calls.
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InDiagnostics,
        crate::Role::Engineer,
        "diagnostics",
    )
    .await;
    super::diagnostics::run(
        std::sync::Arc::new(expect_ticket(store, &id).await),
        ws.clone(),
        job_id.clone(),
    )
    .await;
    assert_eq!(expect_ticket_phase(store, &id).await, TicketPhase::InReview);

    // Record the reviewed base so the reviewer pass is skipped (content-identical).
    let head = crate::git::commands::run_git_head(repo_path.as_path())
        .await
        .ok();
    let tree = crate::git::commands::run_git_write_tree(repo_path.as_path())
        .await
        .ok();
    store
        .set_reviewed_base(&id, head.as_deref(), tree.as_deref())
        .await
        .unwrap();

    // Review: content identical → skip reviewer dispatch (no provider calls).
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InReview,
        crate::Role::Reviewer,
        "review",
    )
    .await;
    super::review::run(
        std::sync::Arc::new(expect_ticket(store, &id).await),
        ws.clone(),
        job_id.clone(),
    )
    .await;
    assert_eq!(expect_ticket_phase(store, &id).await, TicketPhase::InQa);

    // QA: 1 tester, clean verdict → InSanitation.
    let job_id = crate::generate_id();
    spawn_phase_job(&job_id, &ws, &id, TicketPhase::InQa, crate::Role::Qa, "qa").await;
    {
        let fake = fake_clean_verifiers(1);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::qa::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }
    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::InSanitation
    );

    // Dirty the working tree, sanitation inspects and commits → Done.
    let base_head = crate::git::commands::run_git_head(repo_path.as_path())
        .await
        .unwrap();
    std::fs::write(repo_path.join("unexpected.txt"), b"x\n").unwrap();
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InSanitation,
        crate::Role::Sanitation,
        "sanitation",
    )
    .await;
    {
        let fake = crate::util::test::FakeProvider::new()
            .ok("sanitation inspected")
            .ok(r#"{"pass":true,"garbage_files":[],"rationale":"clean"}"#);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::sanitation::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }
    assert_eq!(expect_ticket_phase(store, &id).await, TicketPhase::Done);

    let new_head = crate::git::commands::run_git_head(repo_path.as_path())
        .await
        .unwrap();
    assert_ne!(
        base_head, new_head,
        "a dirty-tree sanitation round must create a new commit",
    );

    drop(repo_dir);
}

// ── 2. Analysis escalation + blocker verification ───────────────────────

/// A base analysis round that flags a shared blocker escalates to 2
/// blocker-verification analysts; the verifiers grade the blocker with
/// substance (kind/severity/impact/reasoning), the ticket still advances to
/// Planning, and the joint comment records the enrichment.

#[serial_test::serial(provider)]
#[tokio::test]
async fn analysis_escalation_and_blocker_verification() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/escalation_ws", "escalation_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Escalation", TicketPhase::Analysis).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::Analysis,
        crate::Role::Analyst,
        "analysis",
    )
    .await;

    {
        // 3 base analysts: 2 flag the SAME blocker (grade=blocker), 1 passes
        // clean. The base consolidation groups both blocker findings; the
        // escalation then runs 2 verifiers that grade the blocker
        // (enrichment-only: one sees it as a risk/edge-case, the other as a
        // main-path blocker).
        let fake = crate::util::test::FakeProvider::new()
            .ok("analyst one")
            .ok(r#"{"issues":[{"text":"missing error handling","grade":"blocker"}]}"#)
            .ok("analyst two")
            .ok(r#"{"issues":[{"text":"missing error handling","grade":"blocker"}]}"#)
            .ok("analyst three")
            .ok(r#"{"issues":[]}"#)
            .ok(r#"{"summary":"both analysts flag missing error handling.","groups":[{"heading":"Missing error handling","contradiction":false,"members":[{"id":0},{"id":1}]}],"ungrouped":[]}"#)
            .ok("verifier one")
            .ok(r#"{"verdicts":[{"index":0,"kind":"risk_edge_case","severity":"medium","impact":"delays onboarding","reasoning":"present but not blocking"}]}"#)
            .ok("verifier two")
            .ok(r#"{"verdicts":[{"index":0,"kind":"main_path_blocker","severity":"high","impact":"blocks the main path","reasoning":"core requirement"}]}"#);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::analysis::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }

    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::Planning,
        "escalation round must still advance to Planning (fail-open)",
    );
    assert!(
        expect_phase_job(store, &id, TicketPhase::Analysis)
            .await
            .is_none(),
    );
    let comments = store.get_comments(&id).await.unwrap();
    let analysis_comment = comments
        .iter()
        .find(|c| c.role == "Analysis")
        .expect("an analysis joint comment must exist");
    assert!(
        analysis_comment.content.contains("Blocker verification"),
        "the joint comment must record the blocker-verification round",
    );
}

// ── 2b. Slot-resume: interrupted parallel-phase round reuses Done slots ──

/// Slot-resume: an interrupted analysis round whose roster already carries a
/// Done slot must reconstruct that slot from its stored outcome and re-run ONLY
/// the not-Done slots with their stored tasks. A fresh rebuild (the pre-fix
/// behavior) would re-run every slot with a new suffix and grow the roster with
/// duplicate-idx rows; here the Done slot is provably NOT re-run (its stored
/// verdict is reconstructed without an LLM call).
#[serial_test::serial(provider)]
#[tokio::test]
async fn analysis_resume_reconstructs_done_slots_and_reruns_not_done() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/resume_analysis_ws", "resume_analysis_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "ResumeAnalysis", TicketPhase::Analysis).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::Analysis,
        crate::Role::Analyst,
        "analysis",
    )
    .await;

    // Seed a partially-completed round: idx 0 Done (a stored clean verdict),
    // idx 1/2 Failed (interrupted before they produced a verdict). The stored
    // per-slot tasks are what the resume re-runs them with.
    let conn = &crate::session::store().conn;
    let done_outcome = super::serialize_verdict_outcome(&super::ParallelVerdict::Analysis(
        crate::AnalysisVerdict {
            issues_detected: Vec::new(),
        },
    ));
    let fail_outcome =
        super::serialize_verdict_outcome(&super::ParallelVerdict::NoResponse("interrupted".into()));
    let seeds = [
        (
            0_i64,
            crate::jobs::RowStatus::Done,
            Some(done_outcome.as_str()),
        ),
        (
            1_i64,
            crate::jobs::RowStatus::Failed,
            Some(fail_outcome.as_str()),
        ),
        (
            2_i64,
            crate::jobs::RowStatus::Failed,
            Some(fail_outcome.as_str()),
        ),
    ];
    for (idx, status, outcome) in seeds {
        let agent_id = format!("ticket_{id}_{idx}_resume_analyst");
        conn.execute(
            crate::jobs::AGENT_INSERT_SQL,
            crate::jobs::agent_params(
                &job_id,
                &agent_id,
                crate::jobs::AgentKind::Analyst,
                Some(idx),
                &format!("resume task {idx}"),
            ),
        )
        .await
        .unwrap();
        crate::jobs::write_agent_outcome(conn, &job_id, &agent_id, status, outcome)
            .await
            .unwrap();
    }

    {
        // Only the 2 not-Done slots are re-run: each consumes a turn response
        // and a verdict extraction. The Done slot is reconstructed (0 calls).
        let fake: std::sync::Arc<crate::util::test::FakeProvider> = std::sync::Arc::new(
            crate::util::test::FakeProvider::new()
                .ok("analyst one")
                .ok(r#"{"issues":[]}"#)
                .ok("analyst two")
                .ok(r#"{"issues":[]}"#),
        );
        let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());
        super::analysis::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
        // 2 re-run slots × (turn + verdict extraction) = 4 LLM calls; a fresh
        // rebuild would have re-run all 3 slots (6 calls).
        assert_eq!(
            fake.request_fingerprints.lock().unwrap().len(),
            4,
            "the Done slot must be reconstructed from its stored outcome, not re-run",
        );
    }

    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::Planning,
        "a resumed analysis round must still advance to Planning (fail-open)",
    );
    assert!(
        expect_phase_job(store, &id, TicketPhase::Analysis)
            .await
            .is_none(),
        "a resumed analysis round that advances must clean up its phase job",
    );
}

/// Slot-resume across the escalation boundary: a base round that already
/// appended its blocker-verification slots is resumed by reconstructing the
/// done escalation slot and re-running only the not-Done one, without
/// re-appending (so no duplicate-idx escalation rows accumulate).
#[serial_test::serial(provider)]
#[tokio::test]
async fn analysis_resume_across_escalation_reuses_done_and_reruns_not_done() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/resume_escalation_ws", "resume_escalation_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "ResumeEscalation", TicketPhase::Analysis).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::Analysis,
        crate::Role::Analyst,
        "analysis",
    )
    .await;

    let conn = &crate::session::store().conn;
    // Base round (idx 0-2): two analysts flag the SAME blocker (grade=blocker),
    // one passes clean. Escalation (idx 3-4): idx 3 already graded the blocker
    // (Done); idx 4 was interrupted (Failed) and must be re-run. The escalation
    // task bakes the blocker list (marker + numbered line) so the resume can
    // re-derive the entries from the stored task.
    let blocker_verdict = super::serialize_verdict_outcome(&super::ParallelVerdict::Analysis(
        crate::AnalysisVerdict {
            issues_detected: vec![crate::AnalysisIssue {
                text: "missing error handling".to_string(),
                grade: crate::IssueGrade::Blocker,
            }],
        },
    ));
    let clean_verdict = super::serialize_verdict_outcome(&super::ParallelVerdict::Analysis(
        crate::AnalysisVerdict {
            issues_detected: Vec::new(),
        },
    ));
    let grade_outcome = super::serialize_verdict_outcome(
        &super::ParallelVerdict::BlockerVerification(crate::BlockerVerificationVerdict {
            verdicts: vec![crate::BlockerVerificationItem {
                index: 0,
                kind: crate::BlockerKind::RiskEdgeCase,
                severity: crate::BlockerSeverity::Low,
                impact: "handled".to_string(),
                reasoning: "handled".to_string(),
            }],
        }),
    );
    let fail_outcome =
        super::serialize_verdict_outcome(&super::ParallelVerdict::NoResponse("interrupted".into()));
    let escalation_task = "The list below is indexed from 0 — report the outcome for each blocker \
         using the matching index.\n\nBlocker list:\n0. missing error handling";
    let seeds = [
        (
            0_i64,
            crate::jobs::RowStatus::Done,
            Some(blocker_verdict.as_str()),
        ),
        (
            1_i64,
            crate::jobs::RowStatus::Done,
            Some(blocker_verdict.as_str()),
        ),
        (
            2_i64,
            crate::jobs::RowStatus::Done,
            Some(clean_verdict.as_str()),
        ),
        (
            3_i64,
            crate::jobs::RowStatus::Done,
            Some(grade_outcome.as_str()),
        ),
        (
            4_i64,
            crate::jobs::RowStatus::Failed,
            Some(fail_outcome.as_str()),
        ),
    ];
    for (idx, status, outcome) in seeds {
        let agent_id = format!("ticket_{id}_{idx}_resume_escalation_analyst");
        conn.execute(
            crate::jobs::AGENT_INSERT_SQL,
            crate::jobs::agent_params(
                &job_id,
                &agent_id,
                crate::jobs::AgentKind::Analyst,
                Some(idx),
                escalation_task,
            ),
        )
        .await
        .unwrap();
        crate::jobs::write_agent_outcome(conn, &job_id, &agent_id, status, outcome)
            .await
            .unwrap();
    }

    {
        // Only the not-Done escalation slot (idx 4) is re-run: turn + verdict
        // extraction. The joint-comment synthesis falls back on the `{}`
        // scripted responses (base verdicts carry issues, so it is not clean).
        let fake = crate::util::test::FakeProvider::new()
            .ok("verifier one")
            .ok(r#"{"verdicts":[{"index":0,"kind":"risk_edge_case","severity":"low","impact":"handled","reasoning":"handled"}]}"#)
            .ok("{}")
            .ok("{}")
            .ok("{}");
        let fake = std::sync::Arc::new(fake);
        let _seam = crate::util::test::install_retry_seam_dyn(fake);
        super::analysis::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }

    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::Planning,
        "a resumed escalation round must still advance to Planning (fail-open)",
    );
    assert!(
        expect_phase_job(store, &id, TicketPhase::Analysis)
            .await
            .is_none(),
    );
    let comments = store.get_comments(&id).await.unwrap();
    let analysis_comment = comments
        .iter()
        .find(|c| c.role == "Analysis")
        .expect("an analysis joint comment must exist");
    assert!(
        analysis_comment.content.contains("Blocker verification"),
        "the joint comment must record the resumed blocker-verification round",
    );
}

#[test]
fn blocker_verification_merge_reduces_two_verifiers_to_one_outcome() {
    let entries = vec![super::analysis::EscalationEntry {
        text: "missing error handling".to_string(),
    }];
    let v1 = crate::BlockerVerificationVerdict {
        verdicts: vec![crate::BlockerVerificationItem {
            index: 0,
            kind: crate::BlockerKind::RiskEdgeCase,
            severity: crate::BlockerSeverity::Medium,
            impact: "delays onboarding".to_string(),
            reasoning: "present but not blocking".to_string(),
        }],
    };
    let v2 = crate::BlockerVerificationVerdict {
        verdicts: vec![crate::BlockerVerificationItem {
            index: 0,
            kind: crate::BlockerKind::MainPathBlocker,
            severity: crate::BlockerSeverity::High,
            impact: "blocks the main path".to_string(),
            reasoning: "core requirement".to_string(),
        }],
    };
    let resolved = super::analysis::apply_blocker_verification(&entries, &[&v1, &v2]);
    assert_eq!(resolved.len(), 1);
    let r = &resolved[0];
    assert_eq!(r.text, "missing error handling");
    assert_eq!(r.kind, crate::BlockerKind::MainPathBlocker);
    assert_eq!(r.severity, crate::BlockerSeverity::High);
    assert_eq!(r.impact, "delays onboarding — blocks the main path");
    assert_eq!(r.reasoning, "present but not blocking — core requirement");
}

// ── 3. Reviewer/QA dynamic count calibration ────────────────────────────

/// The reviewer-count churn bands and P0 floor are product behavior; the QA
/// verifier count is a single tester. These are asserted deterministically
/// without an agent round (real churn is measured for at least one band).

#[serial_test::serial(provider)]
#[tokio::test]
async fn review_qa_dynamic_count_calibration() {
    use crate::pipeline::verdict::{
        DEFAULT_REVIEW_COUNT_HIGH_CHURN, DEFAULT_REVIEW_COUNT_LOW_CHURN,
        DEFAULT_REVIEW_COUNT_TINY_CHURN, review_agent_count, review_base_from_signals,
    };
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let (repo_dir, repo_path) = crate::util::test::init_temp_repo();
    let ws = create_test_workspace(repo_path.to_str().unwrap(), "calib_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Calib", TicketPhase::InReview).await;

    let (tiny, low, high) = (
        DEFAULT_REVIEW_COUNT_TINY_CHURN,
        DEFAULT_REVIEW_COUNT_LOW_CHURN,
        DEFAULT_REVIEW_COUNT_HIGH_CHURN,
    );
    // Literal calibration spec (ticket mahbot-2431): 1–199 → 1 reviewer,
    // 200–999 → 2, 1000–2999 → 3, 3000+ → 4. Each threshold belongs to the
    // higher band.
    assert_eq!(tiny, 200);
    assert_eq!(low, 1000);
    assert_eq!(high, 3000);
    assert_eq!(review_base_from_signals(0, tiny, low, high), 1);
    assert_eq!(review_base_from_signals(199, tiny, low, high), 1);
    assert_eq!(review_base_from_signals(200, tiny, low, high), 2);
    assert_eq!(review_base_from_signals(999, tiny, low, high), 2);
    assert_eq!(review_base_from_signals(1000, tiny, low, high), 3);
    assert_eq!(review_base_from_signals(2999, tiny, low, high), 3);
    assert_eq!(review_base_from_signals(3000, tiny, low, high), 4);
    // P0 floor: priority-0 tickets never drop below 2 reviewers.
    assert_eq!(review_agent_count(1, 0), 2);
    assert_eq!(review_agent_count(2, 0), 2);
    assert_eq!(review_agent_count(4, 0), 4);
    assert_eq!(review_agent_count(1, 5), 1);

    // `compute_reviewer_count` reads real working-tree churn: a 3000+-line
    // untracked file yields the highest band (4 reviewers).
    std::fs::write(repo_path.join("big.txt"), "x\n".repeat(3000)).unwrap();
    let ticket = expect_ticket(store, &id).await;
    let count = crate::pipeline::review::compute_reviewer_count(&ticket, repo_path.as_path()).await;
    assert_eq!(count, 4, "churn >= 3000 must calibrate to 4 reviewers");

    // QA runs exactly one tester per round.
    assert_eq!(crate::pipeline::qa::QA_PARALLEL_AGENT_COUNT, 1);

    // A clean QA round uses exactly 1 verifier (2 responses) → InSanitation.
    drop(repo_dir);
    let ws2 = create_test_workspace("/tmp/qa_calib_ws", "qa_calib_ws").await;
    crate::workspace::store()
        .set_status(&ws2.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let qa_id = make_ticket(store, &ws2, "QaCalib", TicketPhase::InQa).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws2,
        &qa_id,
        TicketPhase::InQa,
        crate::Role::Qa,
        "qa",
    )
    .await;
    {
        let fake = fake_clean_verifiers(1);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::qa::run(
            std::sync::Arc::new(expect_ticket(store, &qa_id).await),
            ws2.clone(),
            job_id.clone(),
        )
        .await;
    }
    assert_eq!(
        expect_ticket_phase(store, &qa_id).await,
        TicketPhase::InSanitation
    );
}

// ── 4. Bounce breaker trips terminal and drains Queued siblings ────────────

/// When the reviewer bounce budget is exhausted the ticket trips to Failed
/// (terminal), the phase job is deleted, Queued siblings are drained to Planning,
/// and the workspace is NOT paused (a bounce is not a technical failure).

#[serial_test::serial(provider)]
#[tokio::test]
async fn bounce_breaker_fails_terminal_and_drains_queued_without_pausing() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let (repo_dir, repo_path) = crate::util::test::init_temp_repo();
    let ws = create_test_workspace(repo_path.to_str().unwrap(), "bounce_trip_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "Trip", TicketPhase::InReview).await;
    let sibling = make_ticket(store, &ws, "Sibling", TicketPhase::Queued).await;

    // Seed the bounce budget at the exhaustion threshold (MAX_BOUNCES = 10).
    store
        .conn
        .execute(
            "UPDATE tickets SET bounce_count = ?1 WHERE id = ?2",
            crate::db::params![10i64, id.clone()],
        )
        .await
        .unwrap();

    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InReview,
        crate::Role::Reviewer,
        "review",
    )
    .await;

    {
        // Zero-change repo → reviewer count 1 → a single sub-threshold verdict
        // trips the breaker (only the agent-turn + extraction are consumed).
        let fake = crate::util::test::FakeProvider::new()
            .ok("reviewer checked")
            .ok(r#"{"score":5,"issues":["bug"]}"#);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::review::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }

    let ticket = expect_ticket(store, &id).await;
    assert_eq!(ticket.phase, TicketPhase::Failed);
    assert_eq!(
        ticket.bounce_count, 10,
        "terminal trip must not consume further bounce budget"
    );
    assert!(
        expect_phase_job(store, &id, TicketPhase::InReview)
            .await
            .is_none(),
    );
    assert_eq!(
        expect_ticket_phase(store, &sibling).await,
        TicketPhase::Planning,
        "breaker trip must drain Queued siblings to Planning",
    );
    assert_workspace_paused(&ws, false).await;

    drop(repo_dir);
}

// ── 5. Hard-failure cleanup + puller re-drive + engineer session stability ──

/// A hard technical failure (no usable output) resets the attempt: the phase
/// job is destroyed, the workspace pause policy is honoured by phase, and the
/// puller re-creates a fresh job. The engineer session pin survives the round
/// job's deletion so the accumulated session is preserved across resets.

#[serial_test::serial(provider)]
#[tokio::test]
async fn reset_round_cleanup_puller_recreates_job_and_engineer_session_stable() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/reset_round_ws", "reset_round_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();

    // ── Analysis hard technical failure ──
    let aid = make_ticket(store, &ws, "HardAnalysis", TicketPhase::Analysis).await;
    let ajob = crate::generate_id();
    spawn_phase_job(
        &ajob,
        &ws,
        &aid,
        TicketPhase::Analysis,
        crate::Role::Analyst,
        "analysis",
    )
    .await;
    {
        // Every analyst turn + every extraction attempt fails to produce a
        // verdict → `extracted_count == 0` → reset for a fresh attempt.
        let fake = crate::util::test::FakeProvider::new()
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json");
        let _seam = crate::util::test::install_retry_seam(fake);
        super::analysis::run(
            std::sync::Arc::new(expect_ticket(store, &aid).await),
            ws.clone(),
            ajob.clone(),
        )
        .await;
    }
    assert_eq!(
        expect_ticket_phase(store, &aid).await,
        TicketPhase::Analysis,
        "a hard analysis failure must stay in Analysis",
    );
    assert!(
        expect_phase_job(store, &aid, TicketPhase::Analysis)
            .await
            .is_none(),
        "a hard analysis failure must destroy the phase job",
    );
    assert_workspace_paused(&ws, false).await;
    {
        let comments = store.get_comments(&aid).await.unwrap();
        assert!(
            comments.iter().any(|c| c
                .content
                .contains("Backlog analysis produced no usable output")),
            "the reset must leave an explanatory comment",
        );
    }

    // The puller re-creates a fresh Analysis job and re-drives it to completion
    // (so no lingering phase-body task consumes the next provider script).
    {
        let fake = fake_clean_verifiers(3);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::dispatch_working_phases(&ws).await;
        assert!(
            expect_phase_job(store, &aid, TicketPhase::Analysis)
                .await
                .is_some(),
            "the puller must re-create a fresh Analysis job after a reset",
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if expect_ticket_phase(store, &aid).await == TicketPhase::Planning
                    && expect_phase_job(store, &aid, TicketPhase::Analysis)
                        .await
                        .is_none()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the re-created analysis round must re-drive to Planning");
    }

    // ── Implementation-phase hard failure pauses + preserves the engine pin ──
    let eid = make_ticket(store, &ws, "HardDev", TicketPhase::InDevelopment).await;
    let ejob = crate::generate_id();
    spawn_phase_job(
        &ejob,
        &ws,
        &eid,
        TicketPhase::InDevelopment,
        crate::Role::Engineer,
        "dev",
    )
    .await;
    let pin_id = crate::jobs::session_pin_id(&eid, crate::Role::Engineer);
    // Seed the engineer's stable session anchor; a hard-failure reset must
    // preserve it across the round job's deletion.
    crate::jobs::upsert_session_pin(
        &crate::session::store().conn,
        &eid,
        "implement",
        crate::jobs::RowStatus::Launched,
        crate::Role::Engineer,
    )
    .await
    .unwrap();
    super::reset_phase_attempt(
        &expect_ticket(store, &eid).await,
        TicketPhase::InDevelopment,
        &ejob,
        "test hard failure",
        "engineer attempt reset",
    )
    .await;
    assert_eq!(
        expect_ticket_phase(store, &eid).await,
        TicketPhase::InDevelopment,
        "an engineer hard failure must leave the ticket in InDevelopment",
    );
    assert!(
        expect_phase_job(store, &eid, TicketPhase::InDevelopment)
            .await
            .is_none(),
    );
    assert_workspace_paused(&ws, true).await;
    let pin_row = crate::session::store()
        .conn
        .query_optional(
            "SELECT 1 FROM agents WHERE agent_id = ?1 AND job_id IS NULL",
            crate::db::params![pin_id.clone()],
            |_| Ok::<_, std::convert::Infallible>(()),
        )
        .await
        .unwrap();
    assert!(
        pin_row.is_some(),
        "the engineer session pin must survive the round job's deletion",
    );

    // The puller re-creates a fresh InDevelopment job on a paused workspace and
    // re-drives it (the re-created body consumes this script).
    {
        let fake = crate::util::test::FakeProvider::new()
            .ok("implemented")
            .ok(r#"{"items":["done"],"summary":"ok"}"#);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::dispatch_working_phases(&ws).await;
        assert!(
            expect_phase_job(store, &eid, TicketPhase::InDevelopment)
                .await
                .is_some(),
            "the puller must re-create a fresh InDevelopment job after a reset",
        );
        // Let the re-created body consume the script and advance the phase.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if expect_ticket_phase(store, &eid).await == TicketPhase::InDiagnostics
                    && expect_phase_job(store, &eid, TicketPhase::InDevelopment)
                        .await
                        .is_none()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the re-created development body must re-drive to InDiagnostics");
    }
}

// ── 6. Analysis hard-failure cleanup stays in phase, no pause ────────────

/// A hard analysis failure (no usable verdict from any analyst) resets the
/// attempt without pausing the workspace or consuming the bounce budget; the
/// ticket stays in Analysis for a fresh round.

#[serial_test::serial(provider)]
#[tokio::test]
async fn analysis_hard_failure_cleanup_stays_in_phase_no_pause() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/hard_analysis_ws", "hard_analysis_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "HardAnalysis", TicketPhase::Analysis).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::Analysis,
        crate::Role::Analyst,
        "analysis",
    )
    .await;

    let before = expect_ticket(store, &id).await;
    {
        let fake = crate::util::test::FakeProvider::new()
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json")
            .ok("not json");
        let _seam = crate::util::test::install_retry_seam(fake);
        super::analysis::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }

    let after = expect_ticket(store, &id).await;
    assert_eq!(
        after.phase,
        TicketPhase::Analysis,
        "ticket must stay in Analysis"
    );
    assert!(
        expect_phase_job(store, &id, TicketPhase::Analysis)
            .await
            .is_none(),
    );
    assert_workspace_paused(&ws, false).await;
    assert_eq!(
        after.bounce_count, before.bounce_count,
        "a hard analysis failure must not consume bounce budget",
    );
    let comments = store.get_comments(&id).await.unwrap();
    assert!(
        comments.iter().any(|c| c
            .content
            .contains("Backlog analysis produced no usable output")),
        "the reset must leave an explanatory comment",
    );
}

// ── 7. GUI cancel brings an in-flight engineer to a terminal Cancelled phase ─

/// The GUI is the single cancel authority: it stops in-flight ticket agents,
/// sets the terminal Cancelled phase (completing the phase job), and pauses the
/// workspace. This holds even while an engineer round is in flight — the
/// agent-side failure tail no longer drives a user cancel.

#[serial_test::serial(provider)]
#[tokio::test]
async fn cancel_requested_engineer_goes_to_cancelled() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/cancel_ws", "cancel_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "CancelDev", TicketPhase::InDevelopment).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InDevelopment,
        crate::Role::Engineer,
        "dev",
    )
    .await;

    {
        // The engineer's first provider outcome is a tool call (empty-args read
        // errors), keeping the agent in-flight until the test cancels it.
        let mut fake = crate::util::test::FakeProvider::new().ok_tool_call("read");
        for _ in 0..9 {
            fake = fake.ok_tool_call("read");
        }
        let _seam = crate::util::test::install_retry_seam(fake);

        let ticket = expect_ticket(store, &id).await;
        let handle = tokio::spawn(super::development::run(
            std::sync::Arc::new(ticket),
            ws.clone(),
            job_id.clone(),
        ));
        wait_for_agent_registered(&id, std::time::Duration::from_secs(2)).await;

        // Reproduce the GUI cancel path: pause the workspace, stop in-flight
        // ticket agents, then transition the ticket to the terminal Cancelled
        // phase (which completes its phase job). The GUI path, not the agent's
        // failure tail, is the source of the terminal state.
        crate::pipeline::pause_workspace_on_failure(
            &expect_ticket(store, &id).await,
            "user cancelled a ticket in the development pipeline",
        )
        .await;
        crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&id);
        store
            .transition_to(&id, None, TicketPhase::Cancelled)
            .await
            .unwrap();
        handle.await.unwrap();
    }

    let ticket = expect_ticket(store, &id).await;
    assert_eq!(
        ticket.phase,
        TicketPhase::Cancelled,
        "a user cancel must move the ticket to a terminal Cancelled phase",
    );
    assert!(
        expect_phase_job(store, &id, TicketPhase::InDevelopment)
            .await
            .is_none(),
    );
    assert_workspace_paused(&ws, true).await;
}

// ── 8. Cooperative pause keeps the job and unpause does not re-pause ─────

/// A cooperative workspace pause freezes an in-flight verifier at its LLM
/// boundary: the phase job is retained, the workspace is paused, and the
/// unpause re-drives the round to completion without re-pausing.

#[serial_test::serial(provider)]
#[tokio::test]
async fn pause_and_resume_keeps_job_and_does_not_re_pause() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/pause_resume_ws", "pause_resume_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    let id = make_ticket(store, &ws, "PauseQa", TicketPhase::InQa).await;
    let job_id = crate::generate_id();
    spawn_phase_job(&job_id, &ws, &id, TicketPhase::InQa, crate::Role::Qa, "qa").await;

    {
        // The QA verifier's first outcome is a tool call (errors), holding it
        // in-flight until the workspace pause lands.
        let mut fake = crate::util::test::FakeProvider::new().ok_tool_call("read");
        for _ in 0..9 {
            fake = fake.ok_tool_call("read");
        }
        let _seam = crate::util::test::install_retry_seam(fake);

        let ticket = expect_ticket(store, &id).await;
        let handle = tokio::spawn(super::qa::run(
            std::sync::Arc::new(ticket),
            ws.clone(),
            job_id.clone(),
        ));
        wait_for_agent_registered(&id, std::time::Duration::from_secs(2)).await;
        crate::workspace::store()
            .set_paused(&ws.name, true)
            .await
            .unwrap();
        handle.await.unwrap();

        assert!(
            expect_phase_job(store, &id, TicketPhase::InQa)
                .await
                .is_some(),
            "a cooperative pause must retain the phase job for the unpause re-drive",
        );
        assert_workspace_paused(&ws, true).await;

        // The interrupted verifier slot is preserved in place (status 'failed'),
        // carrying the stored task the resume re-runs it with — a fresh dispatch
        // would have re-derived a new suffix instead.
        let roster = crate::jobs::list_agents_for_job(&crate::session::store().conn, &job_id)
            .await
            .unwrap();
        assert_eq!(
            roster.len(),
            1,
            "the pause-freeze must preserve the interrupted verifier slot (not clear it)",
        );
        assert_ne!(
            roster[0].status,
            crate::jobs::RowStatus::Launched.as_str(),
            "the paused verifier must not remain marked launched",
        );
    }

    // Resume and re-drive the same round with a clean script: it completes to
    // InSanitation and does NOT re-pause the workspace.
    crate::workspace::store()
        .set_paused(&ws.name, false)
        .await
        .unwrap();
    {
        let fake = fake_clean_verifiers(1);
        let _seam = crate::util::test::install_retry_seam(fake);
        super::qa::run(
            std::sync::Arc::new(expect_ticket(store, &id).await),
            ws.clone(),
            job_id.clone(),
        )
        .await;
    }
    assert_eq!(
        expect_ticket_phase(store, &id).await,
        TicketPhase::InSanitation,
        "the unpause re-drive must complete the QA round",
    );
    assert_workspace_paused(&ws, false).await;
}

/// (i) A stage re-drive whose session carries a dangling analyze call runs the
/// universal resume-completion step BEFORE the engineer's model round: the
/// owned analyze job (Done roster) is terminalized, its consolidated result
/// settles contiguously after the frame, and only then does the engineer LLM
/// round proceed (the Done-slot replay adds zero LLM calls).
#[serial_test::serial(provider)]
#[tokio::test]
async fn stage_re_drive_completes_dangling_calls_before_model_round() {
    let _guard = TEST_LOCK.lock().await;
    init_management_test_stores().await;
    let store = crate::pipeline::board::store();
    let ws = create_test_workspace("/tmp/stage_re_drive_ws", "stage_re_drive_ws").await;
    crate::workspace::store()
        .set_status(&ws.name, &WorkspaceStatus::Ready)
        .await
        .unwrap();
    crate::workspace::store()
        .set_paused(&ws.name, false)
        .await
        .unwrap();

    let id = make_ticket(store, &ws, "StageReDrive", TicketPhase::InDevelopment).await;
    let job_id = crate::generate_id();
    spawn_phase_job(
        &job_id,
        &ws,
        &id,
        TicketPhase::InDevelopment,
        crate::Role::Engineer,
        "implement",
    )
    .await;
    let pin = crate::jobs::session_pin_id(&id, crate::Role::Engineer);
    let conn = &crate::session::store().conn;

    // Seed the stage pin session [user(task), assistant frame with an analyze
    // tool call] + an owned analyze job (Done roster "STAGE_RESULT").
    crate::util::test::seed_session_row(conn, &pin, "user", "implement the ticket").await;
    let frame = crate::providers::reasoning::assistant_replay_payload(
        Some(""),
        &[crate::ToolCall {
            id: "call_stage_analyze".to_string(),
            name: "analyze".to_string(),
            arguments: serde_json::json!({"analyze": "stage analysis"}),
        }],
        None,
    )
    .to_string();
    crate::util::test::seed_session_row(conn, &pin, "assistant", &frame).await;
    let analyze_job = "stage_re_drive_analyze";
    let analyst_id = format!("{analyze_job}_analyst");
    crate::jobs::spawn_job(
        conn,
        analyze_job,
        "stage analysis",
        &ws.name,
        "caller-user",
        "telegram",
        crate::Role::Engineer,
        &[crate::jobs::NewAgent {
            agent_id: analyst_id.clone(),
            kind: crate::jobs::AgentKind::Analyst,
            idx: Some(0),
            task: "stage analysis".to_string(),
        }],
        &crate::jobs::SpawnChild::Analyze,
        Some(&pin),
    )
    .await
    .unwrap();
    crate::jobs::write_agent_outcome(
        conn,
        analyze_job,
        &analyst_id,
        crate::jobs::RowStatus::Done,
        Some("STAGE_RESULT"),
    )
    .await
    .unwrap();

    // Engineer round: 2 responses (answer + summary extraction). The completion
    // (Done-slot replay) adds zero LLM calls, so exactly 2 are consumed.
    let fake = std::sync::Arc::new(
        crate::util::test::FakeProvider::new()
            .ok("implemented the ticket")
            .ok(r#"{"items":["did the thing"],"summary":"done"}"#),
    );
    let _seam = crate::util::test::install_retry_seam_dyn(fake.clone());

    super::development::run(
        std::sync::Arc::new(expect_ticket(store, &id).await),
        ws.clone(),
        job_id.clone(),
    )
    .await;

    // The dangling analyze job was terminalized by the completion (before the
    // engineer's model round).
    let jobs = conn
        .query(
            "SELECT id FROM jobs WHERE id = ?1",
            crate::db::params![analyze_job],
        )
        .await
        .unwrap();
    assert!(
        jobs.is_empty(),
        "the dangling analyze job must be terminalized"
    );

    // The stage session carries the settled tool result contiguous after the
    // frame, then the appended turn message.
    let rows = conn
        .query(
            "SELECT id, role, content FROM sessions WHERE agent_id = ?1 ORDER BY id",
            crate::db::params![pin],
        )
        .await
        .unwrap();
    let roles: Vec<String> = rows.iter().map(|r| r.get::<String>(1).unwrap()).collect();
    assert_eq!(
        roles,
        vec!["user", "assistant", "tool", "user", "assistant"],
        "stage session after completion + turn: {roles:?}"
    );
    let payload: crate::ToolResultPayload =
        serde_json::from_str(&rows[2].get::<String>(2).unwrap()).unwrap();
    assert_eq!(payload.tool_call_id, "call_stage_analyze");
    assert_eq!(payload.content, "STAGE_RESULT");

    // The engineer's model round proceeded AFTER the settle — its first request
    // already carries the settled tool result, and the Done-slot replay added
    // zero LLM calls (exactly the engineer answer + summary extraction ran).
    let messages = fake.request_messages.lock().unwrap();
    assert_eq!(
        messages.len(),
        2,
        "engineer round made exactly two LLM calls"
    );
    assert!(
        messages[0].contains("STAGE_RESULT"),
        "the engineer's first model round sees the settled result"
    );
}
