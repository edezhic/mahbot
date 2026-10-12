//! The analyze tool's cancel action: it stops one of the calling session's
//! dispatched quick analysis rounds, named by the job id its acknowledgement
//! carried, and durably discards its result. The cancel-signal registry lives in
//! [`crate::round_cancel`].
//!
//! Ownership: a round is the caller's while its `jobs` row (or, before that row is
//! written, its in-process cancel-signal entry, which records the dispatching
//! session) does, or while its produced result waits in `pending_jobs`. Anything
//! else is answered as nothing of the caller's by that name — a delivered, an
//! already-cancelled and a never-dispatched id alike.
//!
//! The stop is durable because the round's OWN rows are deleted, never merely
//! signalled: a surviving `kind='analyze'` row would be resumed at the next start
//! and a surviving `pending_jobs` row replayed by the boot replay. The one write
//! that could race that deletion — the dispatch's own row write — is refused by a
//! gate inside its transaction ([`crate::jobs::spawn_job`]) once this cancellation
//! has fired, so a failed deletion is the only way a row outlives the stop, and a
//! caller is then told the cancellation did not happen rather than that it did.
//! The fired signal holds the guarantee at the last moment: an in-tx gate in
//! [`crate::jobs::complete_job_with_envelope`] rolls back a cancelled round's
//! completion and [`crate::tools::SyncDurableCore::stopped_by_cancel`] refuses the
//! route, so a produced-but-undelivered result is never handed over; a copy
//! already on the caller's live channel cannot be recalled — the one accepted
//! race.
//!
//! Only this round's own id is deleted (never by target), its agents stop as a
//! group via [`ParentKey::AnalyzeRound`], and its sessions are left as a completed
//! round's are. A `deep` investigation, a coder round, another agent's round and
//! an id that names nothing are refused by the `kind = 'analyze'` filter and the
//! ownership check and left alone; a repeat cancel of a still-unwinding round is
//! answered as stopped again while its own invocation is still registered — except
//! a round resumed at boot, whose entry records no dispatching session, so once
//! its row is gone the repeat answers like a round that never existed.

use crate::agent::message_router::MessageKind;
use crate::agent::registry::ParentKey;
use crate::db::Connection;

/// What the cancel action did — the three answers a cancel call can give.
#[derive(Debug)]
pub(crate) enum Cancelled {
    /// The round was live and has been stopped.
    Stopped,
    /// The round had already produced its result: the result was discarded and
    /// will never be delivered.
    Discarded,
    /// Nothing of the caller's by that name — the id is not one of the
    /// caller's dispatched quick analyses.
    NotMine,
}

/// Stop `caller`'s round `job_id` and durably discard its result. Ownership,
/// what the answer means and what is left alone are the module doc's subject.
pub(crate) async fn cancel_round(job_id: &str, caller: &str) -> anyhow::Result<Cancelled> {
    let conn = &crate::session::store().conn;
    let Some((outcome, place)) = locate(conn, job_id, caller)
        .await
        .map_err(|e| e.context(not_located(job_id)))?
    else {
        return Ok(Cancelled::NotMine);
    };
    // The signal is fired BEFORE any row is deleted: a completion, a delivery or
    // a dispatch's own row write already in flight must observe it at its last
    // moment — the two in-tx gates read it, and this DELETE serializes behind
    // them whatever they decide.
    crate::round_cancel::cancel(job_id);
    // The round's agents stop as a group, and its consolidation call's
    // observational row goes, so the round stops being shown as running work.
    // Cancellation is cooperative: an in-flight model call may finish, while the
    // round's OWN phase calls (findings extraction and the consolidation pass,
    // see `crate::tools::analyze`) are abandoned at the signal so no further
    // attempt is asked.
    crate::agent::registry::AGENT_REGISTRY
        .cancel_by_parent_key(&ParentKey::AnalyzeRound(job_id.to_string()));
    crate::agent::registry::NON_AGENT_CALLS
        .remove_by_parent_key(&ParentKey::AnalyzeRound(job_id.to_string()));
    // The row removal is what makes the stop durable, so a failure here means
    // the cancellation did NOT happen and the caller is told so — never a claim
    // of a cancellation that was not put in place.
    crate::jobs::delete_round_rows(conn, job_id)
        .await
        .map_err(|e| e.context(not_discarded(job_id)))?;
    // A cancelled round never resumes, so its placed engine goes now — by the
    // directory the row recorded, which names the engine whether or not the
    // directory is still there. A round in a registered folder releases nothing:
    // that name is never the folder's own workspace.
    if let Some(dir) = place {
        crate::tools::delegation::release_placed_engine(&dir);
    }
    tracing::info!(job = %job_id, ?outcome, "Analyze round cancelled");
    Ok(outcome)
}

/// The refusal for a cancel whose round could not be looked up: nothing was
/// changed, so the round carries on untouched. The one answer that never claims
/// a cancellation.
fn not_located(job_id: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "the cancellation of analysis job {job_id} could not be put in place — \
         the round was not touched and carries on"
    )
}

/// The refusal for a cancel whose durable discard could not be committed: the
/// signal is already fired (it cannot be recalled), so this round's own live
/// invocation is cut at its next boundary and its phase calls are abandoned —
/// but the rows survive, and a round whose row survives is resumed by a LATER
/// start of the product, which is the delivery still possible. The cancellation
/// did NOT happen.
fn not_discarded(job_id: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "the cancellation of analysis job {job_id} could not be put in place — its rows could \
         not be removed, so this round's live run delivers nothing but a later start of the \
         product may still deliver its result"
    )
}

/// Find the caller's round in the durable state, in the order that state decides
/// it: the live `jobs` row, then the produced-but-undelivered envelope, then the
/// pre-row cancel-signal entry. The recorded execution directory comes back with
/// the row — only that source has one, and a cancel needs both — so the row is
/// read once. `Ok(None)` is "nothing of the caller's by that name"; an unreadable
/// store is an ERROR, never that answer — the cancel has then not happened and
/// nothing was changed.
async fn locate(
    conn: &Connection,
    job_id: &str,
    caller: &str,
) -> anyhow::Result<Option<(Cancelled, Option<String>)>> {
    // The live round: the row's own delivery address is the ownership proof, and
    // the reader refuses every other `kind` — see
    // [`crate::jobs::analyze_round_owned_by`].
    if let Some(round) = crate::jobs::analyze_round_owned_by(conn, job_id, caller).await? {
        return Ok(Some((Cancelled::Stopped, round.exec_dir)));
    }
    if let Some(envelope) = crate::jobs::pending_envelope(conn, job_id).await? {
        // A queued analyze result addressed to the caller: its round is over and
        // only the delivery is left to discard. The kind filter is load-bearing
        // here too — the same session can have a deep investigation's report or a
        // coder round's report waiting, and those are not this action's to
        // destroy.
        if envelope.kind == MessageKind::AnalyzeToolResult
            && crate::jobs::envelope_target(&envelope) == caller
        {
            return Ok(Some((Cancelled::Discarded, None)));
        }
    }
    // A round whose row is not written yet, or whose row this very cancel just
    // removed while the round unwinds: the dispatching session recorded on its
    // cancel-signal entry is the ownership proof.
    Ok(
        (crate::round_cancel::registered_caller(job_id).as_deref() == Some(caller))
            .then_some((Cancelled::Stopped, None)),
    )
}

/// The cancel action's answer to its caller — one plain line per outcome, kept
/// as inline literals exactly like the sibling tools' dispatch acknowledgements
/// and usage errors.
#[must_use]
pub(crate) fn answer(outcome: &Cancelled, job_id: &str) -> String {
    match outcome {
        Cancelled::Stopped => format!(
            "Cancelled the analysis round (job {job_id}): its analysts stop and its result will \
             not be delivered to this session."
        ),
        Cancelled::Discarded => format!(
            "Analysis round (job {job_id}) had already produced its result — the result was \
             discarded and will not be delivered to this session."
        ),
        Cancelled::NotMine => format!(
            "No analysis round of this session to cancel for job {job_id}: this action stops \
             only the quick analysis rounds this session dispatched, and nothing was changed."
        ),
    }
}
