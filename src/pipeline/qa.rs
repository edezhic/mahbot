//! InQa phase module — the QA tester verifies the change.

use std::sync::Arc;

use crate::Workspace;
use crate::pipeline::board::Ticket;

use super::{QA_VI, TicketPhase, dispatch_verifiers, guard_job_phase};

/// QA runs exactly one tester per round — reviewers already verify the change
/// in depth.
pub(crate) const QA_PARALLEL_AGENT_COUNT: usize = 1;

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, TicketPhase::InQa, &job_id).await {
        return;
    }
    dispatch_verifiers(ticket, ws, QA_VI, job_id).await;
}
