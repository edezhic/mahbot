//! The workspace-stop notice: how a technical death of a stage reaches the workspace's Manager.
//!
//! A technical failure of the development, verification or sanitation stage stops the workspace so
//! the round can be re-run instead of retried blindly. The stop is durable, so the notice is too:
//! one `pause_notices` row per death, written by the death site in the same transaction as the
//! pause it explains ([`stop_workspace`]).
//!
//! The row holds the DEATH, not its text. The notice states the workspace's pause state as it is
//! when it is delivered — a workspace resumed in between is never described as paused — so
//! [`run_delivery_loop`] renders it on every delivery attempt. The two facts it reports are
//! independent: whether THIS failure stopped the workspace (the pause transaction's own answer,
//! frozen with the row) and whether the workspace is paused now (read at delivery).
//!
//! Every death produces its own notice, however often the same stage dies, and two deaths with
//! identical text are two notices: rows are never deduplicated or coalesced. One attempt records
//! one notice (its phase job id is unique), so a stage body that records its death and then panics
//! keeps the notice it already recorded. A notice is consumed only by a Manager turn that ran to
//! completion ([`confirm_delivered`]); a turn that failed or never ran leaves the row and the loop
//! retries it while the service is up. No other durable message changes its guarantee.
//!
//! The notice's prose lives in `src/prompt/pipeline/`: the body, the four pause-state sections and
//! one details asset per cause. Only the failure's data — a participant's label, its
//! classification, its attempt count, its error text — is assembled here.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::agent::message_router::{self, AgentJob, MessageKind};
use crate::db::{self, Row, params};
use crate::pipeline::board::{Ticket, TicketPhase};
use crate::pipeline::verdict::{MemberOutcome, ParallelVerdict};
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::retry::{FailureClass, RetryExhausted};
use crate::workspace;

/// How often the delivery loop looks for a notice that is due.
const TICK: Duration = Duration::from_secs(5);

/// The wait before a notice whose turn did not complete is attempted again. It doubles with every
/// attempt up to [`MAX_RETRY_SECS`], so an outage that outlasts the Manager's turn is retried for
/// as long as the service is up without turning into a hot loop. A notice is never re-routed while
/// the Manager is mid-turn, so this wait only starts once a turn has ended.
const FIRST_RETRY_SECS: i64 = 60;
const MAX_RETRY_SECS: i64 = 1_800;

/// The stage whose death stops the workspace, and therefore notifies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Stage {
    Development,
    Verification,
    Sanitation,
}

impl Stage {
    /// The stage a phase belongs to, when that stage's death stops the workspace. Analysis does
    /// not stop it, so an analysis death has no notice.
    #[must_use]
    pub(crate) fn of_phase(phase: TicketPhase) -> Option<Self> {
        match phase {
            TicketPhase::InDevelopment => Some(Self::Development),
            TicketPhase::Verification => Some(Self::Verification),
            TicketPhase::InSanitation => Some(Self::Sanitation),
            _ => None,
        }
    }

    /// The stage's name as the notice and the Manager's instructions print it.
    fn name(self) -> &'static str {
        match self {
            Self::Development => "Development",
            Self::Verification => "Verification",
            Self::Sanitation => "Sanitation",
        }
    }

    /// The stored form. Never renamed: a row written by an older build must stay readable.
    fn token(self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Verification => "verification",
            Self::Sanitation => "sanitation",
        }
    }

    fn from_token(token: &str) -> Option<Self> {
        [Self::Development, Self::Verification, Self::Sanitation]
            .into_iter()
            .find(|stage| stage.token() == token)
    }
}

/// What died. Serialized into the row as written, so a notice replayed after a restart reports
/// the failure exactly as it was.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum Cause {
    /// The engineer's hard failure — the notice the development stage already had.
    Engineer { terms: Terms },
    /// The sanitation round could not finish: the repository state was unreadable or the commit
    /// failed — the notice the sanitation stage already had.
    SanitationStop { terms: Terms },
    /// Round participants never produced a usable result. One notice per death, naming every
    /// participant that failed.
    Participants(Vec<Member>),
    /// The workspace's project commands could not be loaded — the round cannot be judged without
    /// them, and no participant was involved.
    ProjectCommands { terms: Terms },
    /// The stage body crashed.
    Panic(String),
}

/// One member of a round that failed.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Member {
    /// Its name in the round — "code reviewer 2", "functional tester", "sanitation agent".
    pub label: String,
    /// What it produced instead of a result.
    pub produced: Produced,
    pub terms: Terms,
}

impl Member {
    /// The member a round reports as a stage death, or `None` when it completed or was
    /// cut short — a phase-gate bail, a pause-freeze, a cancellation, a shutdown or a drain.
    #[must_use]
    pub(crate) fn of(label: &str, member: &MemberOutcome) -> Option<Self> {
        if member.cut_short {
            return None;
        }
        let (produced, terms) = match &member.verdict {
            ParallelVerdict::NoResponse(reason) => (
                Produced::NoAnswer,
                Terms::new(
                    member.class,
                    member.attempts,
                    crate::util::failure_detail(reason, "stage failure"),
                ),
            ),
            ParallelVerdict::ParseFailed(exhausted) => {
                (Produced::UnreadableVerdict, Terms::of_extraction(exhausted))
            }
            ParallelVerdict::Verdict(_)
            | ParallelVerdict::Analysis(_)
            | ParallelVerdict::BlockerVerification(_) => return None,
        };
        Some(Self {
            label: label.to_string(),
            produced,
            terms,
        })
    }
}

/// What a failed participant produced instead of a usable result.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) enum Produced {
    /// It never produced an answer.
    NoAnswer,
    /// It produced an answer whose verdict could not be read.
    UnreadableVerdict,
}

impl Produced {
    /// The index of this outcome's shape in `stage_failure_participants.md`, counting from the
    /// intro section, in the asset's order.
    fn section(self) -> usize {
        match self {
            Self::NoAnswer => 0,
            Self::UnreadableVerdict => 1,
        }
    }
}

/// How a failure is reported: the product's own classification, the attempts the failing call
/// made before giving up, and the error text as recorded — masked and capped once, when the row
/// is written ([`crate::util::failure_detail`]), so the row is exactly what the notice says.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Terms {
    /// The classification label, `None` when the product has no classification for this kind of
    /// failure — reported as such rather than invented.
    pub class: Option<String>,
    /// `None` when the product cannot say how many attempts the failing call made.
    pub attempts: Option<usize>,
    pub error: String,
}

impl Terms {
    #[must_use]
    pub(crate) fn new(
        class: Option<FailureClass>,
        attempts: Option<usize>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            class: class.map(|class| class.label().to_string()),
            attempts,
            error: error.into(),
        }
    }

    /// A failure the product does not classify — a store read, a crashed body, a missing answer.
    #[must_use]
    pub(crate) fn plain(error: impl Into<String>) -> Self {
        Self::new(None, None, error)
    }

    /// A verdict extraction that ran out of attempts: its class, how many attempts it made and
    /// the last attempt's failure chain.
    #[must_use]
    pub(crate) fn of_extraction(exhausted: &RetryExhausted) -> Self {
        Self::new(
            Some(exhausted.final_class),
            Some(exhausted.failures.len()),
            crate::util::failure_detail(&exhausted.detail, "verdict extraction failure"),
        )
    }
}

/// One attempt records one notice: the phase job id is unique, so a stage body that reports its
/// death and then panics keeps the notice it already recorded.
///
/// The same uniqueness also coalesces one narrow case: a second genuine death of an attempt that a
/// drain resumed under the SAME job id, while that attempt's earlier notice is still undelivered.
/// It needs a drain to start between the pause commit and the engineer's re-check, and a notice
/// already recorded for that attempt is the more truthful one either way.
const INSERT_NOTICE: &str = "INSERT INTO pause_notices \
     (id, workspace_name, ticket_id, ticket_title, stage, cause, stopped_workspace, job_id, \
      created_at, attempted_at, delivery_attempts) \
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, NULL, 0) \
     ON CONFLICT(job_id) DO NOTHING";

/// Record a stage's technical death and stop the workspace for it, in ONE transaction: a pause is
/// never committed without the notice that explains it, so a crash between the two cannot leave a
/// frozen workspace with nobody told. A workspace already paused by something else is not stopped
/// by this failure, and the row says so.
///
/// A death cut short by a shutdown or drain is not a failure and does nothing. A failure here is
/// logged and never propagates into the stage's own failure path.
pub(crate) async fn stop_workspace(ticket: &Ticket, stage: Stage, cause: Cause, job_id: &str) {
    if crate::shutdown::aborting() {
        return;
    }
    let Ok(cause) = serde_json::to_string(&cause) else {
        error!(
            ticket = %ticket.id,
            stage = stage.token(),
            "Failed to serialize a workspace-stop notice's cause — the notice is lost",
        );
        return;
    };
    let recorded = crate::workspace::store()
        .pause_with_record(&ticket.workspace_name, async |tx, stopped| {
            let inserted = tx
                .execute(
                    INSERT_NOTICE,
                    params![
                        crate::generate_id(),
                        ticket.workspace_name.clone(),
                        ticket.id.clone(),
                        ticket.title.clone(),
                        stage.token().to_string(),
                        cause.clone(),
                        i64::from(stopped),
                        job_id.to_string(),
                        db::now(),
                    ],
                )
                .await?;
            if inserted == 0 {
                debug!(
                    ticket = %ticket.id,
                    job = %job_id,
                    "This attempt already recorded its workspace-stop notice",
                );
            } else {
                info!(
                    ticket = %ticket.id,
                    workspace = %ticket.workspace_name,
                    stage = stage.token(),
                    stopped_workspace = stopped,
                    "Recorded a workspace-stop notice",
                );
            }
            Ok(())
        })
        .await;
    match recorded {
        Ok(Some(_)) => {}
        Ok(None) => warn!(
            ticket = %ticket.id,
            workspace = %ticket.workspace_name,
            "A stage death names no managed workspace — no pause and nobody to tell",
        ),
        Err(e) => error!(
            ticket = %ticket.id,
            workspace = %ticket.workspace_name,
            stage = stage.token(),
            error = %e,
            "Failed to stop the workspace for a stage death — no pause and no notice",
        ),
    }
}

/// Remove one row: a notice a completed turn consumed, an orphan, or one no build can read.
async fn delete(notice_id: &str) -> anyhow::Result<()> {
    crate::session::store()
        .conn
        .execute(
            "DELETE FROM pause_notices WHERE id = ?1",
            params![notice_id.to_string()],
        )
        .await?;
    Ok(())
}

/// A Manager turn that ran to completion consumed the notice — whether or not it said anything in
/// that turn. A turn that failed or never ran does not call this, so its row stays and the loop
/// retries it.
pub(crate) async fn confirm_delivered(notice_id: &str) {
    if let Err(e) = delete(notice_id).await {
        warn!(notice = %notice_id, error = %e, "Failed to confirm a workspace-stop notice's delivery");
    }
}

/// The notice delivery loop, for the process's whole life (spawned by the boot's background set).
pub async fn run_delivery_loop() {
    loop {
        deliver_due().await;
        if !crate::shutdown::sleep_or_shutdown_or_drain(TICK).await {
            return;
        }
    }
}

/// Route every notice that is due; a drain stops it from routing a fresh Manager turn while the
/// process winds down.
async fn deliver_due() {
    if crate::shutdown::aborting() {
        return;
    }
    let rows = match undelivered().await {
        Ok(rows) => rows,
        Err(e) => {
            warn!(error = %e, "Failed to read the undelivered workspace-stop notices");
            return;
        }
    };
    let now = chrono::Utc::now();
    for row in rows.iter().filter(|row| is_due(row, now)) {
        deliver_row(row).await;
    }
}

/// Route one notice, with the workspace's pause state read NOW.
async fn deliver_row(row: &Notice) {
    // The workspace's Manager is mid-turn: its own turn may still complete with this notice, and
    // routing again would only queue a duplicate behind it. The row stays due, so it goes out as
    // soon as the turn ends.
    if crate::agent::registry::AGENT_REGISTRY
        .contains(&crate::address::AgentAddress::manager(&row.workspace_name).id())
    {
        return;
    }
    // Stamped before the read and the route: a crash or a failed read between them costs a retry
    // after the backoff, never a duplicate on the next tick.
    if let Err(e) = stamp_attempt(&row.id).await {
        warn!(notice = %row.id, error = %e, "Failed to stamp a workspace-stop notice's delivery attempt");
        return;
    }
    match workspace::store().get_by_name(&row.workspace_name).await {
        Ok(Some(ws)) => route(row, ws.paused),
        // Nothing to tell a workspace that no longer exists, and nothing will ever deliver this
        // row: it goes, rather than retrying forever. `WorkspaceStore::delete` already removes a
        // deleted workspace's notices with the workspace itself, so this only catches a row an
        // older build left behind.
        Ok(None) => {
            warn!(
                notice = %row.id,
                workspace = %row.workspace_name,
                "A workspace-stop notice names a workspace that no longer exists — dropping it",
            );
            if let Err(e) = delete(&row.id).await {
                warn!(notice = %row.id, error = %e, "Failed to drop an orphaned workspace-stop notice");
            }
        }
        Err(e) => warn!(
            notice = %row.id,
            workspace = %row.workspace_name,
            error = %e,
            "Failed to read a workspace's pause state — the notice waits for the next attempt",
        ),
    }
}

/// Whether the notice is due for another delivery attempt: never attempted, or the backoff that
/// follows its attempt count has elapsed.
fn is_due(row: &Notice, now: chrono::DateTime<chrono::Utc>) -> bool {
    let Some(attempted) = row.attempted_at.as_deref() else {
        return true;
    };
    let Ok(attempted) = db::parse_utc_timestamp(attempted) else {
        return true;
    };
    attempted + retry_delay(row.delivery_attempts) <= now
}

/// The wait after `attempts` failed deliveries: doubling from [`FIRST_RETRY_SECS`] to
/// [`MAX_RETRY_SECS`].
fn retry_delay(attempts: u32) -> chrono::TimeDelta {
    let doublings = attempts.saturating_sub(1).min(5);
    let secs = FIRST_RETRY_SECS.saturating_mul(1_i64 << doublings);
    chrono::TimeDelta::try_seconds(secs.min(MAX_RETRY_SECS)).unwrap_or_default()
}

async fn stamp_attempt(notice_id: &str) -> anyhow::Result<()> {
    crate::session::store()
        .conn
        .execute(
            "UPDATE pause_notices SET attempted_at = ?2, delivery_attempts = delivery_attempts + 1 \
             WHERE id = ?1",
            params![notice_id.to_string(), db::now()],
        )
        .await?;
    Ok(())
}

/// Hand the rendered notice to the workspace's Manager. The row id rides the envelope so the
/// consumer confirms exactly this notice when the turn completes.
fn route(row: &Notice, paused_now: bool) {
    let content = render(row, paused_now);
    message_router::route(
        &crate::address::AgentAddress::manager(&row.workspace_name).id(),
        AgentJob {
            content,
            workspace_name: row.workspace_name.clone(),
            user_name: "system".to_string(),
            channel: String::new(),
            kind: MessageKind::UserMessage,
            role: crate::Role::Manager,
            reply_target: None,
            pending_job_id: None,
            pause_notice_id: Some(row.id.clone()),
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        },
    );
}

/// One undelivered notice, as stored.
struct Notice {
    id: String,
    workspace_name: String,
    ticket_id: String,
    ticket_title: String,
    stage: Stage,
    cause: Cause,
    stopped_workspace: bool,
    attempted_at: Option<String>,
    delivery_attempts: u32,
}

crate::columns! {
    NOTICE_COLUMNS [NOTICE] {
        ID                => "id",
        WORKSPACE_NAME    => "workspace_name",
        TICKET_ID         => "ticket_id",
        TICKET_TITLE      => "ticket_title",
        STAGE             => "stage",
        CAUSE             => "cause",
        STOPPED_WORKSPACE => "stopped_workspace",
        ATTEMPTED_AT      => "attempted_at",
        DELIVERY_ATTEMPTS => "delivery_attempts",
    }
}

async fn undelivered() -> anyhow::Result<Vec<Notice>> {
    let rows = crate::session::store()
        .conn
        .query(
            &format!("SELECT {NOTICE_COLUMNS} FROM pause_notices ORDER BY created_at"),
            (),
        )
        .await?;
    let mut notices = Vec::with_capacity(rows.len());
    for row in &rows {
        match notice_row(row) {
            Ok(notice) => notices.push(notice),
            // No build can render this row, so nothing will ever deliver it: it goes, rather than
            // being re-read and re-warned on every tick.
            Err(e) => {
                warn!(error = %e, "Unreadable workspace-stop notice row — dropping it");
                if let Ok(id) = row.get::<String>(COL_NOTICE_ID)
                    && let Err(e) = delete(&id).await
                {
                    warn!(notice = %id, error = %e, "Failed to drop an unreadable workspace-stop notice");
                }
            }
        }
    }
    Ok(notices)
}

/// A stored notice, or the reason this build cannot read it — never delivered with an invented
/// cause.
fn notice_row(row: &Row) -> anyhow::Result<Notice> {
    let stage_token: String = row.get(COL_NOTICE_STAGE)?;
    let stage = Stage::from_token(&stage_token)
        .ok_or_else(|| anyhow::anyhow!("unknown stage '{stage_token}'"))?;
    Ok(Notice {
        id: row.get(COL_NOTICE_ID)?,
        workspace_name: row.get(COL_NOTICE_WORKSPACE_NAME)?,
        ticket_id: row.get(COL_NOTICE_TICKET_ID)?,
        ticket_title: row.get(COL_NOTICE_TICKET_TITLE)?,
        stage,
        cause: serde_json::from_str(&row.get::<String>(COL_NOTICE_CAUSE)?)?,
        stopped_workspace: row.get::<i64>(COL_NOTICE_STOPPED_WORKSPACE)? != 0,
        attempted_at: row.get(COL_NOTICE_ATTEMPTED_AT)?,
        delivery_attempts: u32::try_from(row.get::<i64>(COL_NOTICE_DELIVERY_ATTEMPTS)?)
            .unwrap_or(0),
    })
}

fn render(row: &Notice, paused_now: bool) -> String {
    let details = details(row);
    substitute(
        &load_prompt("pipeline/workspace_stop_notification.md"),
        &[
            ("{{stage}}", row.stage.name()),
            ("{{workspace_name}}", &row.workspace_name),
            ("{{ticket_id}}", &row.ticket_id),
            ("{{ticket_title}}", &row.ticket_title),
            // The assets end with their last line's newline; the body's blank lines are its own.
            ("{{failure_details}}", details.trim()),
            (
                "{{pause_facts}}",
                &pause_facts(row.stopped_workspace, paused_now),
            ),
        ],
    )
}

/// The notice's two independent facts — whether this failure stopped the workspace, and whether
/// the workspace is paused right now — in the asset's section order.
fn pause_facts(stopped_workspace: bool, paused_now: bool) -> String {
    let sections = load_prompt_sections("pipeline/workspace_stop_pause_facts.md");
    let index = match (stopped_workspace, paused_now) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    };
    // The asset carries exactly these four sections (a test holds it to that); an empty one
    // costs the notice a paragraph, never its delivery.
    sections.get(index).cloned().unwrap_or_default()
}

/// What failed and how, for the row's cause: one details asset per cause, only the data filled in.
fn details(row: &Notice) -> String {
    match &row.cause {
        Cause::Engineer { terms } => {
            with_terms(&load_prompt("pipeline/stage_failure_engineer.md"), terms)
        }
        Cause::SanitationStop { terms } => with_terms(
            &load_prompt("pipeline/stage_failure_sanitation_stop.md"),
            terms,
        ),
        Cause::ProjectCommands { terms } => with_terms(
            &load_prompt("pipeline/stage_failure_project_commands.md"),
            terms,
        ),
        Cause::Participants(participants) => participants_block(participants),
        Cause::Panic(message) => substitute(
            &load_prompt("pipeline/stage_failure_panic.md"),
            &[("{{terms}}", &terms_block(&Terms::plain(message)))],
        ),
    }
}

/// A details asset with its `{{terms}}` filled in.
fn with_terms(template: &str, terms: &Terms) -> String {
    substitute(template, &[("{{terms}}", &terms_block(terms))])
}

/// The failed participants: the list's intro, then one bullet per participant — the asset's shape
/// for what it produced, with the product's own terms indented under it.
fn participants_block(participants: &[Member]) -> String {
    let sections = load_prompt_sections("pipeline/stage_failure_participants.md");
    let bullets = participants
        .iter()
        .map(|participant| {
            // Section 0 is the intro; the shapes follow in `Produced` order.
            let shape = sections
                .get(1 + participant.produced.section())
                .cloned()
                .unwrap_or_default();
            substitute(
                &shape,
                &[
                    ("{{label}}", &participant.label),
                    ("{{terms}}", &indent(&terms_block(&participant.terms))),
                ],
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let intro = sections.first().cloned().unwrap_or_default();
    substitute(&intro, &[("{{participants}}", &bullets)])
}

/// How a failure is reported: the product's classification (or that it has none), the attempts
/// the failing call made before giving up, and the error text as recorded. The lines are
/// `stage_failure_terms.md`'s four sections, in its order: classification, no classification,
/// attempts, error.
fn terms_block(terms: &Terms) -> String {
    let sections = load_prompt_sections("pipeline/stage_failure_terms.md");
    let line = |index: usize, replacements: &[(&str, &str)]| {
        sections
            .get(index)
            .map(|section| substitute(section, replacements))
            .unwrap_or_default()
    };
    let mut lines = vec![match &terms.class {
        Some(class) => line(0, &[("{{class}}", class)]),
        None => line(1, &[]),
    }];
    if let Some(attempts) = terms.attempts {
        lines.push(line(2, &[("{{attempts}}", &attempts.to_string())]));
    }
    lines.push(line(3, &[("{{error}}", terms.error.trim())]));
    lines.retain(|line| !line.is_empty());
    lines.join("\n")
}

/// Every line of `text` indented two spaces — a participant's terms sit under its bullet.
fn indent(text: &str) -> String {
    text.lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Record a notice directly, for tests that need a row without driving a death site.
#[cfg(test)]
pub(crate) async fn record_for_test(id: &str, workspace: &str, job_id: &str, cause: &Cause) {
    crate::session::store()
        .conn
        .execute(
            INSERT_NOTICE,
            params![
                id.to_string(),
                workspace.to_string(),
                "t-1".to_string(),
                "A title".to_string(),
                Stage::Development.token().to_string(),
                serde_json::to_string(cause).expect("serialize a notice's cause"),
                0_i64,
                job_id.to_string(),
                db::now(),
            ],
        )
        .await
        .expect("record a notice");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{
        STORE_LOCK, TicketFixture, create_test_workspace, init_management_test_stores,
        init_test_stores,
    };

    fn ticket(workspace: &str) -> Ticket {
        TicketFixture::new("mahbot-1", TicketPhase::Verification)
            .title("A stage that dies must say why")
            .workspace(workspace)
            .build()
    }

    /// A notice as the delivery loop reads one.
    fn notice(stage: Stage, cause: Cause, stopped_workspace: bool) -> Notice {
        Notice {
            id: "n1".to_string(),
            workspace_name: "mahbot".to_string(),
            ticket_id: "mahbot-1".to_string(),
            ticket_title: "A stage that dies must say why".to_string(),
            stage,
            cause,
            stopped_workspace,
            attempted_at: None,
            delivery_attempts: 0,
        }
    }

    /// The rows one test's workspace owns — the shared test store keeps every other test's.
    async fn stored(workspace: &str) -> Vec<Notice> {
        undelivered()
            .await
            .expect("read the recorded notices")
            .into_iter()
            .filter(|row| row.workspace_name == workspace)
            .collect()
    }

    /// The assets the notice is assembled from by section index: a missing section would silently
    /// drop a line of the notice instead of failing loudly.
    #[test]
    fn the_sectioned_assets_carry_every_section_the_renderer_chooses() {
        assert_eq!(
            load_prompt_sections("pipeline/workspace_stop_pause_facts.md").len(),
            4,
            "one pause-state section per combination of the two facts"
        );
        assert_eq!(
            load_prompt_sections("pipeline/stage_failure_terms.md").len(),
            4,
            "classification, no classification, attempts, error"
        );
        assert_eq!(
            load_prompt_sections("pipeline/stage_failure_participants.md").len(),
            3,
            "the list intro and one participant shape per outcome"
        );
    }

    /// Every combination of the two independent facts reads differently, and each paused one
    /// asks for the relay while neither promises who lifts the pause.
    #[test]
    fn the_four_pause_facts_are_distinct_and_truthful() {
        let combinations = [(true, true), (true, false), (false, true), (false, false)];
        let facts: Vec<String> = combinations
            .into_iter()
            .map(|(stopped, paused)| pause_facts(stopped, paused))
            .collect();
        for (index, fact) in facts.iter().enumerate() {
            assert!(!fact.is_empty(), "section {index} must not be empty");
        }
        let unique: std::collections::HashSet<&String> = facts.iter().collect();
        assert_eq!(unique.len(), 4, "each combination states its own facts");
        for (stopped, paused) in combinations {
            let fact = pause_facts(stopped, paused);
            assert_eq!(
                fact.contains("did not stop the workspace"),
                !stopped,
                "the notice must state whether this failure stopped the workspace: {fact}"
            );
            assert_eq!(
                fact.contains("is paused right now"),
                paused,
                "the notice must state whether the workspace is paused now: {fact}"
            );
            assert_eq!(
                fact.contains("You cannot resume the workspace yourself"),
                paused,
                "only a paused workspace has a resume to relay: {fact}"
            );
            assert!(
                !fact.contains("No action is needed"),
                "a paused workspace always needs the relay: {fact}"
            );
        }
    }

    /// The notice names the workspace, the ticket by id and title, the stage, every failed
    /// participant, and how each one failed in the product's own terms.
    #[test]
    fn a_participant_death_names_every_failed_participant() {
        let row = notice(
            Stage::Verification,
            Cause::Participants(vec![
                Member {
                    label: "code reviewer 1".to_string(),
                    produced: Produced::NoAnswer,
                    terms: Terms::new(
                        Some(FailureClass::Transport),
                        Some(13),
                        "13 attempt(s) failed (last: transport): connection reset",
                    ),
                },
                Member {
                    label: "functional tester".to_string(),
                    produced: Produced::UnreadableVerdict,
                    terms: Terms::plain("no verdict in the answer"),
                },
            ]),
            true,
        );

        let text = render(&row, true);

        assert!(text.starts_with("⚠️ Workspace stop"), "{text}");
        for needle in [
            "mahbot",
            "mahbot-1",
            "A stage that dies must say why",
            "Verification",
            "code reviewer 1 — produced no answer",
            "functional tester — produced an answer but its verdict could not be read",
            "Classification: transport.",
            "Attempts before giving up: 13.",
            "Error: 13 attempt(s) failed (last: transport): connection reset",
            "The product recorded no classification for this failure.",
            "Error: no verdict in the answer",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
        assert!(!text.contains("No participant was involved"), "{text}");
    }

    /// A death with no participant says what failed instead and that nobody took part.
    #[test]
    fn a_project_commands_death_names_what_failed_and_that_no_participant_was_involved() {
        let row = notice(
            Stage::Verification,
            Cause::ProjectCommands {
                terms: Terms::plain("I/O error (tempdir)"),
            },
            false,
        );

        let text = render(&row, false);

        for needle in [
            "The workspace's project commands could not be loaded",
            "no participant was involved",
            "The product recorded no classification for this failure.",
            "Error: I/O error (tempdir)",
            "did not stop the workspace",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }

    /// Only the three stages whose death stops the workspace notify, and those are exactly the
    /// pipeline-occupied phases: a new occupied phase must gain a stage (or a phase a stage) or
    /// this test fails, so a pause and its notice can never drift apart.
    #[test]
    fn only_the_three_stages_whose_death_stops_the_workspace_notify() {
        use strum::IntoEnumIterator;
        for (phase, stage) in [
            (TicketPhase::InDevelopment, Some(Stage::Development)),
            (TicketPhase::Verification, Some(Stage::Verification)),
            (TicketPhase::InSanitation, Some(Stage::Sanitation)),
            (TicketPhase::Analysis, None),
        ] {
            assert_eq!(Stage::of_phase(phase), stage, "{phase:?}");
        }
        for phase in TicketPhase::iter() {
            assert_eq!(
                Stage::of_phase(phase).is_some(),
                phase.is_pipeline_occupied(),
                "a stage's death stops a workspace exactly when the phase occupies the pipeline: {phase:?}",
            );
        }
    }

    /// A crashed stage body is reported as a crash carrying the panic message, and the notice says
    /// the product has no classification for it rather than inventing one.
    #[tokio::test]
    async fn a_crashed_stage_body_is_reported_as_a_crash() {
        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        create_test_workspace("/tmp/notice_crash_ws", "notice_crash_ws").await;
        let ticket = ticket("notice_crash_ws");

        stop_workspace(
            &ticket,
            Stage::Sanitation,
            Cause::Panic("index out of bounds".to_string()),
            "crash_job",
        )
        .await;

        let rows = stored("notice_crash_ws").await;
        assert_eq!(rows.len(), 1, "the crash records one notice");
        let text = render(&rows[0], true);
        for needle in [
            "Sanitation",
            "crashed",
            "index out of bounds",
            "The product recorded no classification for this failure.",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }

    /// One death is one row and two deaths are two rows, however alike: nothing is deduplicated
    /// or coalesced.
    #[tokio::test]
    async fn identical_deaths_are_separate_notices() {
        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        create_test_workspace("/tmp/notice_identical_ws", "notice_identical_ws").await;
        let ticket = ticket("notice_identical_ws");
        let cause = || Cause::Panic("boom".to_string());

        stop_workspace(&ticket, Stage::Development, cause(), "death_a").await;
        // The workspace is resumed in between, so both deaths are the ones that stopped it and
        // the two notices read identically.
        workspace::store()
            .set_paused("notice_identical_ws", false)
            .await
            .expect("resume the test workspace");
        stop_workspace(&ticket, Stage::Development, cause(), "death_b").await;

        let rows = stored("notice_identical_ws").await;
        assert_eq!(rows.len(), 2, "two deaths are two notices");
        assert_ne!(rows[0].id, rows[1].id);
        assert_eq!(render(&rows[0], true), render(&rows[1], true));
    }

    /// One attempt records one notice: a stage body that records its death and then panics keeps
    /// the notice it already recorded.
    #[tokio::test]
    async fn one_attempt_records_one_notice() {
        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        create_test_workspace("/tmp/notice_one_ws", "notice_one_ws").await;
        let ticket = ticket("notice_one_ws");

        stop_workspace(
            &ticket,
            Stage::Development,
            Cause::Panic("boom".to_string()),
            "one_job",
        )
        .await;
        stop_workspace(
            &ticket,
            Stage::Development,
            Cause::Panic("boom".to_string()),
            "one_job",
        )
        .await;

        assert_eq!(
            stored("notice_one_ws").await.len(),
            1,
            "the second record of the same attempt is a no-op"
        );
    }

    /// A workspace already paused by something else was not stopped by THIS failure, and the row
    /// says so.
    #[tokio::test]
    async fn a_death_in_an_already_paused_workspace_says_it_did_not_stop_it() {
        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        let ws = create_test_workspace("/tmp/notice_paused_ws", "notice_paused_ws").await;
        workspace::store()
            .set_paused(&ws.name, true)
            .await
            .expect("pause the test workspace");
        let ticket = ticket("notice_paused_ws");

        stop_workspace(
            &ticket,
            Stage::Verification,
            Cause::Panic("boom".to_string()),
            "paused_job",
        )
        .await;

        let rows = stored("notice_paused_ws").await;
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].stopped_workspace);
        let text = render(&rows[0], true);
        assert!(text.contains("did not stop the workspace"), "{text}");
        assert!(text.contains("is paused right now"), "{text}");
    }

    /// Deleting a workspace removes the notices it will never be told, so nothing is left behind
    /// to retry against a name that no longer exists.
    #[tokio::test]
    async fn deleting_a_workspace_drops_its_undelivered_notices() {
        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        create_test_workspace("/tmp/notice_delete_ws", "notice_delete_ws").await;
        let ticket = ticket("notice_delete_ws");
        stop_workspace(
            &ticket,
            Stage::Development,
            Cause::Panic("boom".to_string()),
            "delete_job",
        )
        .await;
        assert_eq!(stored("notice_delete_ws").await.len(), 1);

        workspace::store()
            .delete("notice_delete_ws")
            .await
            .expect("delete the test workspace");

        assert!(
            stored("notice_delete_ws").await.is_empty(),
            "a deleted workspace's notices go with it"
        );
    }

    /// Delivery confirmation is the only thing that consumes a notice: the row survives until a
    /// Manager turn completes with it.
    #[tokio::test]
    async fn a_delivered_notice_is_consumed_and_an_undelivered_one_stays() {
        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        create_test_workspace("/tmp/notice_consume_ws", "notice_consume_ws").await;
        let ticket = ticket("notice_consume_ws");
        stop_workspace(
            &ticket,
            Stage::Sanitation,
            Cause::Panic("boom".to_string()),
            "consume_job",
        )
        .await;
        let id = stored("notice_consume_ws").await[0].id.clone();

        confirm_delivered(&id).await;

        assert!(
            stored("notice_consume_ws").await.is_empty(),
            "a completed turn consumes it"
        );
        confirm_delivered(&id).await;
        assert!(
            stored("notice_consume_ws").await.is_empty(),
            "confirming twice is harmless"
        );
    }

    /// The retry wait doubles with the attempt count and never exceeds the cap, so an outage
    /// outlasting the Manager's turn cannot turn the loop into a hot one; a stamp makes the notice
    /// due again only after that wait.
    #[tokio::test]
    async fn the_retry_wait_doubles_up_to_its_cap_and_a_stamp_waits_for_it() {
        let secs = |attempts| retry_delay(attempts).num_seconds();
        assert_eq!(secs(1), FIRST_RETRY_SECS);
        assert_eq!(secs(2), FIRST_RETRY_SECS * 2);
        assert_eq!(secs(3), FIRST_RETRY_SECS * 4);
        assert_eq!(secs(20), MAX_RETRY_SECS);

        let _guard = STORE_LOCK.lock().await;
        init_test_stores().await;
        create_test_workspace("/tmp/notice_backoff_ws", "notice_backoff_ws").await;
        let ticket = ticket("notice_backoff_ws");
        stop_workspace(
            &ticket,
            Stage::Development,
            Cause::Panic("boom".to_string()),
            "backoff_job",
        )
        .await;
        let id = stored("notice_backoff_ws").await[0].id.clone();
        assert!(
            is_due(&stored("notice_backoff_ws").await[0], chrono::Utc::now()),
            "never attempted"
        );

        stamp_attempt(&id).await.unwrap();

        let row = &stored("notice_backoff_ws").await[0];
        assert!(
            !is_due(row, chrono::Utc::now()),
            "the backoff has not elapsed"
        );
        assert!(
            is_due(
                row,
                chrono::Utc::now()
                    + retry_delay(row.delivery_attempts)
                    + chrono::TimeDelta::seconds(1)
            ),
            "the backoff elapses"
        );
    }

    /// The loop hands the notice to the workspace's Manager, carrying its own row id so the
    /// consumer confirms exactly this notice, and states the pause state read at delivery.
    #[tokio::test]
    async fn delivery_routes_the_notice_with_the_pause_state_read_now() {
        let _guard = STORE_LOCK.lock().await;
        init_management_test_stores().await;
        let ws = create_test_workspace("/tmp/notice_delivery_ws", "notice_delivery_ws").await;
        assert!(!ws.paused, "a fresh test workspace is not paused");
        let ticket = ticket("notice_delivery_ws");
        stop_workspace(
            &ticket,
            Stage::Verification,
            Cause::Panic("boom".to_string()),
            "delivery_job",
        )
        .await;
        let row = &stored("notice_delivery_ws").await[0];
        let id = row.id.clone();
        // Resumed between the failure and the delivery: the notice still reports the failure as
        // the one that stopped the workspace, and the pause state as it is NOW.
        workspace::store()
            .set_paused("notice_delivery_ws", false)
            .await
            .expect("resume the test workspace");

        let mut inbox = message_router::register_agent(
            &crate::address::AgentAddress::manager("notice_delivery_ws").id(),
        );

        deliver_row(row).await;

        let job = inbox
            .try_recv()
            .expect("the notice reaches the workspace's Manager");
        assert_eq!(job.role, crate::Role::Manager);
        assert_eq!(job.pause_notice_id.as_deref(), Some(id.as_str()));
        assert!(
            job.content.contains("stopped the workspace"),
            "{}",
            job.content
        );
        assert!(
            job.content.contains("is not paused now"),
            "the state is read at delivery, not frozen at the failure: {}",
            job.content
        );

        // A stamped notice waits for its backoff instead of flooding the Manager.
        let row = &stored("notice_delivery_ws").await[0];
        assert!(
            !is_due(row, chrono::Utc::now()),
            "the delivery attempt starts the retry wait"
        );
        assert!(inbox.try_recv().is_err());
        assert_eq!(stored("notice_delivery_ws").await.len(), 1);
        message_router::unregister_agent(
            &crate::address::AgentAddress::manager("notice_delivery_ws").id(),
        );
    }

    /// A round frozen by a pause, a superseded round and a completed participant are never
    /// reported as deaths; a genuine technical failure is.
    #[test]
    fn only_a_genuine_technical_failure_is_a_stage_death() {
        let member = |verdict, paused, cut_short| MemberOutcome {
            verdict,
            paused,
            cut_short,
            class: Some(FailureClass::Transport),
            attempts: Some(3),
        };
        let failure = || ParallelVerdict::NoResponse("provider down".to_string());

        let reported = Member::of("functional tester", &member(failure(), false, false))
            .expect("a technical failure is a death");
        assert_eq!(reported.label, "functional tester");
        assert_eq!(reported.terms.attempts, Some(3));
        assert!(
            Member::of("functional tester", &member(failure(), true, true)).is_none(),
            "a pause-frozen round is not a death"
        );
        assert!(
            Member::of("functional tester", &member(failure(), false, true)).is_none(),
            "a cancelled or superseded round is not a death"
        );
        assert!(
            Member::of(
                "functional tester",
                &MemberOutcome::of_verdict(ParallelVerdict::Verdict(crate::Verdict {
                    score: 9,
                    issues_detected: Vec::new(),
                })),
            )
            .is_none(),
            "a completed participant is not a death"
        );
    }

    /// The Manager's instructions name the class of notice by the same opening the asset prints,
    /// so the class the prompt recognises and the class the product sends cannot drift apart.
    #[test]
    fn the_manager_instructions_name_the_notice_by_its_opening() {
        let notice = load_prompt("pipeline/workspace_stop_notification.md");
        let heading = notice
            .lines()
            .next()
            .expect("the notice opens with a heading");
        let opening = heading
            .split(" — ")
            .next()
            .expect("the heading opens with the class's name");
        assert!(opening.starts_with("⚠️ Workspace stop"), "{heading}");
        assert!(
            load_prompt("role/manager.md").contains(opening),
            "the Manager's instructions must name the notice by `{opening}`",
        );
    }
}
