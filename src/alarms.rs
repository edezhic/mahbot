//! Alarm/reminder persistence and fire routing for the Assistant session.
//!
//! Backs the `add_alarm` / `list_alarms` / `remove_alarm` tools and the
//! periodic background sweep that routes due reminders back into the
//! Assistant's own personal session as user-role messages.
//!
//! An alarm optionally carries a [`StoredTrigger`] — one custom tool plus its
//! arguments, stored in the `trigger` column. At fire time the tool runs as the
//! alarm's owner in the owner's personal workspace, and only what the check
//! reported decides whether the Assistant is woken at all. A stored check that
//! cannot be read is a check that cannot run, and is treated as one.

use std::collections::HashSet;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Context as _;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration};

use crate::Workspace;
use crate::db;
use crate::tools::custom::CallRefusal;
use crate::tools::shell::ProgramOutcome;
use crate::util::UnwrapPoison;

crate::define_store! {
    pub(crate) static ALARMS: AlarmStore,
    expect = "ALARMS not initialized — call init_all_stores() first",
}

// ── Column index constants ────────────────────────────────────────────────

crate::columns! {
    ALARM_COLUMNS [ALARM] {
        ID               => "id",
        SESSION_ID       => "session_id",
        USER_NAME        => "user_name",
        KIND             => "kind",
        TEXT             => "text",
        TRIGGER          => "trigger",
        INTERVAL_SECONDS => "interval_seconds",
        NEXT_FIRE_AT     => "next_fire_at",
    }
}

/// A single alarm/reminder row.
#[derive(Debug, Clone)]
pub(crate) struct Alarm {
    pub id: String,
    pub session_id: String,
    /// The raw (un-escaped) user the alarm belongs to.
    pub user_name: String,
    /// `"one-shot"` or `"periodic"`.
    pub kind: String,
    pub text: String,
    /// Optional custom-tool check to run at fire time (triggered alarm).
    pub trigger: Option<StoredTrigger>,
    /// Periodic interval in seconds.
    pub interval_seconds: Option<i64>,
    /// RFC3339 UTC next fire time.
    pub next_fire_at: String,
}

/// An alarm's trigger: one custom tool, plus the arguments that tool receives.
///
/// The tool is named exactly as a `custom` call names it and must be one the
/// alarm's owner can call; one tool per alarm — no chains. Arguments are stored
/// as the tool's declared parameters, already coerced to their declared types.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Trigger {
    pub tool: String,
    /// Defaulted so a row written without the key still decodes; the product
    /// always stores the object.
    #[serde(default)]
    pub args: serde_json::Map<String, serde_json::Value>,
}

impl Trigger {
    /// The stored form: `{"tool": <name>, "args": {…}}`.
    fn encode(&self) -> String {
        serde_json::to_string(self).expect("a trigger is always serializable")
    }

    /// The trigger as it is shown: the tool's name followed by its arguments,
    /// nothing but the name when it has none.
    fn render(&self) -> String {
        if self.args.is_empty() {
            return self.tool.clone();
        }
        format!(
            "{} {}",
            self.tool,
            serde_json::to_string(&self.args)
                .expect("a trigger's arguments are always serializable")
        )
    }
}

/// The `trigger` column's content for one alarm: the check that was stored
/// there, or a value that cannot be read as one.
///
/// A stored check that cannot be read is a check that cannot run — it wakes the
/// assistant with the reason and takes the alarm with it, rather than quietly
/// becoming a plain reminder that keeps firing with its check dropped.
#[derive(Debug, Clone)]
pub(crate) enum StoredTrigger {
    /// The decoded check: the tool to run and the arguments it receives.
    Tool(Trigger),
    /// A value in the column that is not a stored check.
    Unreadable,
}

/// What the notice and the `<user-alarms>` summary show in place of a stored
/// check that cannot be read: there is no tool name to print.
const UNREADABLE_TRIGGER: &str = "(unreadable stored check)";

impl StoredTrigger {
    /// Decode a stored `trigger` column value. Only this module writes the
    /// column, so a value that does not decode is not a state the product
    /// produces — it reads as [`Self::Unreadable`] all the same, because the
    /// fire path's rule is about what can be read, not about how the row came
    /// to hold what it holds.
    fn decode(raw: &str) -> Self {
        match serde_json::from_str(raw) {
            Ok(trigger) => Self::Tool(trigger),
            Err(_) => Self::Unreadable,
        }
    }

    /// How a stored trigger is shown to the assistant — the confirmation when
    /// it is armed, the `list_alarms` line and the `<user-alarms>` summary:
    /// the tool's name followed by its arguments, nothing but the name when it
    /// has none, and a marker when there is nothing readable to show.
    ///
    /// The two tool-output surfaces are ordinary tool results, so the
    /// agent-level pass scrubs them like any other; the `<user-alarms>` summary
    /// and the fired notice carry the alarm's own wording and are not scrubbed.
    pub(crate) fn render(&self) -> String {
        match self {
            Self::Tool(trigger) => trigger.render(),
            Self::Unreadable => UNREADABLE_TRIGGER.to_string(),
        }
    }
}

fn alarm_from_row(row: &db::Row) -> Result<Alarm, ::turso::Error> {
    Ok(Alarm {
        id: row.get(COL_ALARM_ID)?,
        session_id: row.get(COL_ALARM_SESSION_ID)?,
        user_name: row.get(COL_ALARM_USER_NAME)?,
        kind: row.get(COL_ALARM_KIND)?,
        text: row.get(COL_ALARM_TEXT)?,
        trigger: stored_trigger(row)?,
        interval_seconds: row.get(COL_ALARM_INTERVAL_SECONDS)?,
        next_fire_at: row.get(COL_ALARM_NEXT_FIRE_AT)?,
    })
}

/// Read a row's `trigger` column: `None` when it is NULL, otherwise the stored
/// check. Only text can be the stored check this product writes, so a row that
/// holds anything else reads as [`StoredTrigger::Unreadable`] — a check that
/// cannot be read is a check that cannot run either, and must not fail the
/// sweep or leave the alarm firing with its check dropped.
fn stored_trigger(row: &db::Row) -> Result<Option<StoredTrigger>, ::turso::Error> {
    match row.get_value(COL_ALARM_TRIGGER)? {
        db::Value::Null => Ok(None),
        db::Value::Text(raw) => Ok(Some(StoredTrigger::decode(&raw))),
        _ => Ok(Some(StoredTrigger::Unreadable)),
    }
}

/// Maximum number of active alarms allowed per session.
const MAX_ACTIVE_ALARMS: i64 = 10;

/// Upper bound on an alarm's whole stored trigger — the tool's name together
/// with its arguments — in characters. Keeps an over-eagerly-large trigger from
/// bloating the notification envelope or the `<user-alarms>` prompt summary.
const MAX_ALARM_TRIGGER_CHARS: usize = 2000;

/// Upper bound on a periodic interval (~292 years): keeps the total swept
/// advance (interval × skipped whole periods) within a representable
/// [`DateTime`] range, so the O(1) periodic skip saturates instead of
/// overflowing. Also rejects absurd intervals as caller errors at creation.
const MAX_PERIOD_SECS: i64 = i64::MAX / 1_000_000_000;

/// Format an RFC3339 timestamp for display as `"{local} local time ({utc} UTC)"`.
pub(crate) fn format_fire_time(timestamp: &str) -> Result<String> {
    let dt = DateTime::parse_from_rfc3339(timestamp)?;
    let local = dt
        .with_timezone(&chrono::Local)
        .format("%Y-%m-%d %H:%M:%S %Z");
    let utc = dt.with_timezone(&chrono::Utc).format("%Y-%m-%d %H:%M:%S");
    Ok(format!("{local} local time ({utc} UTC)"))
}

// ── Alarm CRUD ──────────────────────────────────────────────────────────

/// Create a new alarm for a session.
///
/// Exactly one of `fire_at` (one-shot) or `interval_seconds` (periodic) must be
/// provided. One-shot fire times are normalized to RFC3339 UTC and must be in
/// the future; periodic intervals must be at least 5 seconds. At most
/// [`MAX_ACTIVE_ALARMS`] active alarms may exist per session.
pub(crate) async fn add_alarm(
    session_id: &str,
    user_name: &str,
    text: &str,
    fire_at: Option<&str>,
    interval_seconds: Option<u64>,
    trigger: Option<Trigger>,
) -> Result<Alarm> {
    // Validate an optional trigger before touching the DB: the whole stored
    // trigger — the tool's name together with its arguments — is bounded.
    let encoded_trigger = trigger.as_ref().map(Trigger::encode);
    if let Some(encoded) = &encoded_trigger {
        anyhow::ensure!(
            encoded.chars().count() <= MAX_ALARM_TRIGGER_CHARS,
            "Alarm trigger too long (maximum {MAX_ALARM_TRIGGER_CHARS} characters)"
        );
    }

    // Normalize the one-shot fire time to RFC3339 UTC.
    let normalized_fire_at = fire_at
        .map(|f| {
            db::parse_utc_timestamp(f)
                .map(|dt| dt.to_rfc3339())
                .with_context(|| format!("Invalid RFC3339/ISO-8601 fire time: {f}"))
        })
        .transpose()?;

    // Exactly one of fire_at / interval_seconds; a one-shot must be in the
    // future (a past timestamp is a caller error, never a fire-on-next-sweep).
    let (kind, interval_secs, next_fire_at) = match (&normalized_fire_at, interval_seconds) {
        (Some(fire), None) => {
            anyhow::ensure!(
                db::parse_utc_timestamp(fire)? > db::parse_utc_timestamp(&db::now())?,
                "Cannot set an alarm for a time in the past"
            );
            ("one-shot", None, fire.clone())
        }
        (None, Some(interval)) => {
            anyhow::ensure!(interval >= 5, "Period must be at least 5 seconds");
            let interval_secs = i64::try_from(interval)
                .ok()
                .filter(|v| *v <= MAX_PERIOD_SECS)
                .with_context(|| "Period must be at most 292 years")?;
            // Periodic: first fire one interval from now (no immediate fire, no
            // backlog payout — the sweep skips missed whole periods thereafter).
            // Saturate an absurd interval to a far-future fire time rather than
            // panicking the caller on a DateTime overflow.
            let start = db::parse_utc_timestamp(&db::now())?;
            let next = start
                .checked_add_signed(ChronoDuration::seconds(interval_secs))
                .unwrap_or(DateTime::<chrono::Utc>::MAX_UTC);
            ("periodic", Some(interval_secs), next.to_rfc3339())
        }
        _ => anyhow::bail!("Exactly one of fire_at or interval_seconds must be provided"),
    };

    // Enforce the per-session active cap.
    let rows = store()
        .conn
        .query(
            "SELECT COUNT(*) FROM alarms WHERE session_id = ?1 AND status = 'active'",
            db::params![session_id],
        )
        .await?;
    let count: i64 = rows.first().map(|r| r.get(0)).transpose()?.unwrap_or(0);
    anyhow::ensure!(
        count < MAX_ACTIVE_ALARMS,
        "Alarm limit reached (maximum 10 active alarms)"
    );

    let id = crate::generate_id();
    let created_at = db::now();
    store()
        .conn
        .execute(
            "INSERT INTO alarms \
             (id, session_id, user_name, kind, text, fire_at, interval_seconds, next_fire_at, status, created_at, trigger) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'active', ?9, ?10)",
            db::params![
                id.clone(),
                session_id,
                user_name,
                kind,
                text,
                normalized_fire_at.clone(),
                interval_secs,
                next_fire_at.clone(),
                created_at.clone(),
                encoded_trigger.clone(),
            ],
        )
        .await?;

    Ok(Alarm {
        id,
        session_id: session_id.to_string(),
        user_name: user_name.to_string(),
        kind: kind.to_string(),
        text: text.to_string(),
        trigger: trigger.map(StoredTrigger::Tool),
        interval_seconds: interval_secs,
        next_fire_at,
    })
}

/// List the active alarms for a session, ordered by next fire time.
pub(crate) async fn list_alarms(session_id: &str) -> Result<Vec<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE session_id = ?1 AND status = 'active' \
         ORDER BY next_fire_at ASC"
    );
    let rows = store().conn.query(&sql, db::params![session_id]).await?;
    Ok(rows.iter().map(alarm_from_row).collect::<Result<_, _>>()?)
}

/// List the active alarms for a user across all their sessions, ordered by
/// next fire time. Unlike [`list_alarms`] (session-scoped) this is the
/// user-wide view backing the Assistant's system-prompt alarm snapshot.
pub(crate) async fn list_user_alarms(user_name: &str) -> Result<Vec<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE user_name = ?1 AND status = 'active' \
         ORDER BY next_fire_at ASC"
    );
    let rows = store().conn.query(&sql, db::params![user_name]).await?;
    Ok(rows.iter().map(alarm_from_row).collect::<Result<_, _>>()?)
}

/// Every active alarm of every owner, ordered by owner name and then by next
/// fire time — the read backing the read-only alarms page. The page groups by
/// the name recorded on the alarm itself, so an alarm whose account is gone is
/// still listed under it.
pub(crate) async fn list_all_active_alarms() -> Result<Vec<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE status = 'active' \
         ORDER BY user_name ASC, next_fire_at ASC"
    );
    let rows = store().conn.query(&sql, db::params![]).await?;
    Ok(rows.iter().map(alarm_from_row).collect::<Result<_, _>>()?)
}

/// Mark an active alarm as removed; returns the removed alarm or `None`.
pub(crate) async fn remove_alarm(session_id: &str, id: &str) -> Result<Option<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE id = ?1 AND session_id = ?2 AND status = 'active'"
    );
    let rows = store()
        .conn
        .query(&sql, db::params![id, session_id])
        .await?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let alarm = alarm_from_row(row)?;
    store()
        .conn
        .execute(
            "UPDATE alarms SET status = 'removed' WHERE id = ?1 AND session_id = ?2",
            db::params![id, session_id],
        )
        .await?;
    Ok(Some(alarm))
}

/// Soft-delete an alarm regardless of its current status (idempotent).
///
/// Unlike [`remove_alarm`] this has no `status = 'active'` gate, so it also
/// deletes an already-fired one-shot. Used whenever a trigger run did not come
/// back clean — a failed check and a check that could not run alike — so a
/// broken alarm can never re-fire.
async fn delete_alarm_after_run(alarm: &Alarm) -> Result<()> {
    store()
        .conn
        .execute(
            "UPDATE alarms SET status = 'removed' WHERE id = ?1 AND session_id = ?2",
            db::params![alarm.id.as_str(), alarm.session_id.as_str()],
        )
        .await?;
    Ok(())
}

/// The due alarms at `now` (RFC3339 UTC), ordered by next fire time.
async fn due_alarms(now: &str) -> Result<Vec<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE status = 'active' AND next_fire_at <= ?1 \
         ORDER BY next_fire_at ASC"
    );
    let rows = store().conn.query(&sql, db::params![now]).await?;
    Ok(rows.iter().map(alarm_from_row).collect::<Result<_, _>>()?)
}

// ── Fire / sweep ────────────────────────────────────────────────────────

/// Route a due alarm into the Assistant's own session and update its state.
///
/// A triggered alarm (`trigger` set) runs its custom tool and delivers the
/// check's result; a plain alarm delivers the reminder text. The notification is
/// delivered as a user-role message into the calling assistant's own session
/// (`alarm.session_id`, resolved directly — never re-derived from the id, which
/// would double-escape colliding user names). One-shot alarms are terminalized
/// (`status='fired'`); periodic alarms advance `next_fire_at` past every missed
/// whole period.
async fn fire_alarm(alarm: &Alarm) -> Result<()> {
    match &alarm.trigger {
        Some(trigger) => fire_trigger_alarm(alarm, trigger).await,
        None => fire_plain_alarm(alarm).await,
    }
}

/// Deliver a plain (untriggered) alarm reminder and advance its state.
///
/// The durable envelope is persisted BEFORE advancing state (see
/// [`deliver_alarm_notification`]): a crash after persisting but before the
/// state advance replays the reminder at boot and leaves the alarm due, so the
/// reminder is never lost.
async fn fire_plain_alarm(alarm: &Alarm) -> Result<()> {
    let now = db::now();
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("alarm_notification.md"),
        &[
            ("{{text}}", &alarm.text),
            ("{{fire_at}}", &alarm.next_fire_at),
        ],
    );
    deliver_alarm_notification(alarm, content).await;
    advance_alarm_state(alarm, &now).await?;
    Ok(())
}

/// Deliver an alarm's notice into the calling assistant's session, logging a
/// lost durable copy against the alarm it came from.
///
/// Delivery is sourced from the stored raw user and session id so the notice
/// targets the right personal session regardless of agent-ID escaping — never
/// re-derived from the id, which would double-escape colliding user names.
async fn deliver_alarm_notification(alarm: &Alarm, content: String) {
    if let Err(e) = crate::agent::message_router::deliver_assistant_notice(
        &alarm.session_id,
        &alarm.user_name,
        content,
    )
    .await
    {
        tracing::warn!(
            alarm = %alarm.id,
            error = %e,
            "Failed to persist alarm delivery — routing best-effort"
        );
    }
}

/// Advance an alarm's stored state past `now`: one-shot → `status='fired'`;
/// periodic → `next_fire_at` past every missed whole period.
async fn advance_alarm_state(alarm: &Alarm, now: &str) -> Result<()> {
    if alarm.kind == "one-shot" {
        store()
            .conn
            .execute(
                "UPDATE alarms SET status = 'fired' WHERE id = ?1",
                db::params![alarm.id.as_str()],
            )
            .await?;
    } else {
        let interval = alarm.interval_seconds.unwrap_or(0);
        anyhow::ensure!(
            interval > 0,
            "Periodic alarm {} has no valid interval",
            alarm.id
        );
        let next = next_periodic_fire(now, &alarm.next_fire_at, interval)?;
        store()
            .conn
            .execute(
                "UPDATE alarms SET next_fire_at = ?1 WHERE id = ?2",
                db::params![next, alarm.id.as_str()],
            )
            .await?;
    }
    Ok(())
}

/// Alarms whose trigger check is currently in flight, keyed by alarm id — the
/// periodic-overlap guard that prevents a slow check from piling up one spawn
/// per sweep.
static TRIGGERS_IN_FLIGHT: LazyLock<std::sync::Mutex<HashSet<String>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

/// Try to mark an alarm id's check run as in flight: `true` claims it (the
/// spawned task's [`InFlightGuard`] releases it), `false` means a previous run
/// of the same alarm's check is still active.
fn claim_in_flight(alarm_id: &str) -> bool {
    TRIGGERS_IN_FLIGHT
        .lock()
        .unwrap_poison()
        .insert(alarm_id.to_string())
}

/// RAII guard that removes an alarm id from [`TRIGGERS_IN_FLIGHT`] on drop, so
/// every exit path (including a panic/abort of the spawned check task) clears
/// the in-flight marker and lets the next firing run.
struct InFlightGuard(String);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        TRIGGERS_IN_FLIGHT.lock().unwrap_poison().remove(&self.0);
    }
}

/// Fire a triggered alarm: advance state, then spawn a detached task that runs
/// the stored check and delivers its result. Every non-clean outcome also
/// soft-deletes the alarm (see [`run_alarm_trigger_task`]); a stored check that
/// cannot be read never runs at all and takes that same path.
async fn fire_trigger_alarm(alarm: &Alarm, trigger: &StoredTrigger) -> Result<()> {
    // Advance state FIRST — independent of the check's outcome — so a slow or
    // failing check cannot keep the alarm due and re-fire it on every sweep.
    advance_alarm_state(alarm, &db::now()).await?;

    // Periodic-overlap guard: if a previous run of this alarm is still in
    // flight, skip this firing entirely (advanced but no spawn, no notification).
    if !claim_in_flight(&alarm.id) {
        return Ok(());
    }

    tokio::spawn(run_alarm_trigger_task(alarm.clone(), trigger.clone()));
    Ok(())
}

/// What one trigger evaluation produced, as the notification sees it.
enum TriggerOutcome {
    /// The check ran and reported nothing at all — the alarm stays armed and
    /// nothing is delivered.
    Silent,
    /// The check ran and reported something: the run's own outcome.
    Ran(ProgramOutcome),
    /// The check could not run at all: the reason the assistant is shown. The
    /// alarm goes either way.
    CouldNotRun(String),
}

impl TriggerOutcome {
    /// The wake decision over one finished run: only a clean run that reported
    /// nothing at all stays silent. Everything else wakes the assistant — a run
    /// that failed, and output of any kind, whatever it looks like.
    fn from_run(outcome: ProgramOutcome) -> Self {
        if outcome.success && !outcome.has_output {
            Self::Silent
        } else {
            Self::Ran(outcome)
        }
    }

    /// Whether the alarm stays armed: the check ran and exited cleanly.
    fn clean(&self) -> bool {
        match self {
            Self::Silent => true,
            Self::Ran(outcome) => outcome.success,
            Self::CouldNotRun(_) => false,
        }
    }
}

/// Run one trigger: resolve the named custom tool for the alarm's owner — the
/// same availability rule, catalogue lookup and argument contract a normal
/// `custom` call applies, with the strict rule that arguments the tool does not
/// declare are refused rather than ignored — then run it in the owner's
/// personal workspace through the managed runtime.
///
/// Nothing is credential-scrubbed here: what the check produced is what the
/// assistant is shown, and the run's raw streams alone drive the wake decision
/// ([`TriggerOutcome::from_run`]).
async fn run_trigger(ws: &Workspace, user: &str, trigger: &Trigger) -> TriggerOutcome {
    let call =
        match crate::tools::custom::resolve_tool_call(user, &trigger.tool, &trigger.args, true)
            .await
        {
            Ok(call) => call,
            Err(CallRefusal::Unavailable { .. }) => {
                return TriggerOutcome::CouldNotRun(format!(
                    "the custom tool \"{}\" is no longer available to you",
                    trigger.tool
                ));
            }
            Err(CallRefusal::Name { .. } | CallRefusal::NotUsable { .. }) => {
                return TriggerOutcome::CouldNotRun(format!(
                    "the custom tool \"{}\" is no longer usable",
                    trigger.tool
                ));
            }
            Err(CallRefusal::Arguments(e)) => {
                return TriggerOutcome::CouldNotRun(format!(
                    "the stored arguments no longer fit \"{}\" — {e}",
                    trigger.tool
                ));
            }
        };

    let Some(bun) = crate::tools::bun::bun_binary_path() else {
        return TriggerOutcome::CouldNotRun("the managed bun runtime is unavailable".to_string());
    };
    TriggerOutcome::from_run(
        crate::tools::shell::run_program_outcome(
            ws,
            &bun,
            &[call.path.display().to_string(), call.payload()],
        )
        .await,
    )
}

/// Run a triggered alarm's stored check in the owner's personal workspace and
/// deliver its result — or stay silent when the check reported nothing at all.
///
/// Every non-clean outcome soft-deletes the alarm before the notification is
/// delivered, so a broken check can never keep the alarm armed and re-firing.
async fn run_alarm_trigger_task(alarm: Alarm, trigger: StoredTrigger) {
    // The in-flight marker is removed unconditionally on drop.
    let _in_flight = InFlightGuard(alarm.id.clone());

    let outcome = match &trigger {
        StoredTrigger::Tool(check) => {
            let ws = crate::users::personal_workspace_struct(&alarm.user_name);
            run_trigger(&ws, &alarm.user_name, check).await
        }
        // A stored check that cannot be read is a check that cannot run:
        // nothing is run, and the reason goes to the assistant with the alarm.
        StoredTrigger::Unreadable => {
            TriggerOutcome::CouldNotRun("the stored check cannot be read".to_string())
        }
    };

    // Track whether the delete actually committed — the notification must not
    // claim a deletion that didn't happen.
    let deletion = if outcome.clean() {
        None
    } else {
        Some(delete_alarm_after_run(&alarm).await)
    };

    match trigger_notification(&alarm, &trigger, &outcome, deletion.as_ref()) {
        Some(content) => deliver_alarm_notification(&alarm, content).await,
        None => {
            tracing::debug!(alarm = %alarm.id, "alarm check reported nothing — staying silent");
        }
    }
}

/// Decide the delivery for a finished trigger evaluation: `None` stays silent
/// (the check ran and reported nothing at all); `Some` is the rendered
/// `<alarm-notification>` with the reminder, the trigger, the status line, the
/// deletion note (on every non-clean outcome) and the check's output.
/// `deletion` carries the result of [`delete_alarm_after_run`], `None` on
/// the clean path.
fn trigger_notification(
    alarm: &Alarm,
    trigger: &StoredTrigger,
    outcome: &TriggerOutcome,
    deletion: Option<&Result<()>>,
) -> Option<String> {
    let (status, output) = match outcome {
        TriggerOutcome::Silent => return None,
        TriggerOutcome::Ran(outcome) => {
            let status = if outcome.success {
                format!("The check exited successfully ({}).", outcome.detail)
            } else {
                format!("The check FAILED ({}).", outcome.detail)
            };
            let shown = if !outcome.output.is_empty() {
                crate::util::truncate_sandwich(
                    &outcome.output,
                    crate::util::TOOL_OUTPUT_BUDGET_BYTES,
                    "check output",
                )
            } else if outcome.has_output {
                // Escape-only output: reported, so it woke the assistant, but
                // nothing readable survives the ANSI strip — the notice must not
                // call that "no output".
                "(output was not printable)".to_string()
            } else {
                "(no output)".to_string()
            };
            (status, format!("\n\n{shown}"))
        }
        TriggerOutcome::CouldNotRun(reason) => {
            (format!("The check could not run: {reason}."), String::new())
        }
    };
    let deletion_line = match deletion {
        None => String::new(),
        Some(Ok(())) => crate::prompt::load_prompt("alarm_trigger_deletion.md"),
        Some(Err(_)) => crate::prompt::load_prompt("alarm_trigger_deletion_failed.md"),
    };
    Some(crate::prompt::substitute(
        &crate::prompt::load_prompt("alarm_trigger_notification.md"),
        &[
            ("{{text}}", &alarm.text),
            ("{{fire_at}}", &alarm.next_fire_at),
            ("{{trigger}}", &trigger.render()),
            ("{{trigger_status}}", &status),
            ("{{trigger_deletion}}", &deletion_line),
            ("{{trigger_output}}", &output),
        ],
    ))
}

/// Compute the next fire time for a periodic alarm: advance `next_fire` past
/// `now` by whole periods (O(1), saturating on overflow). Never loops over
/// downtime and never panics on an absurd interval.
fn next_periodic_fire(now: &str, next_fire: &str, interval_secs: i64) -> Result<String> {
    let now_dt = db::parse_utc_timestamp(now)?;
    let next = db::parse_utc_timestamp(next_fire)?;
    let elapsed = now_dt.signed_duration_since(next).num_seconds();
    let to_skip = elapsed.div_euclid(interval_secs).saturating_add(1);
    // Cap the single advance (interval × skipped periods) so it stays within a
    // representable DateTime range and cannot overflow/panic the sweep.
    let add_secs = interval_secs.saturating_mul(to_skip).min(MAX_PERIOD_SECS);
    let next = next
        .checked_add_signed(ChronoDuration::seconds(add_secs))
        .unwrap_or(DateTime::<chrono::Utc>::MAX_UTC);
    Ok(next.to_rfc3339())
}

/// Fire up to `batch_limit` due alarms, continuing past individual failures so
/// one bad alarm cannot block the rest of the sweep.
async fn run_alarm_sweep(batch_limit: usize) -> Result<()> {
    let due = due_alarms(&db::now()).await?;
    let mut fired = 0usize;
    let mut failed = 0usize;
    for alarm in due.into_iter().take(batch_limit) {
        match fire_alarm(&alarm).await {
            Ok(()) => fired += 1,
            Err(e) => {
                failed += 1;
                tracing::warn!(alarm = %alarm.id, error = %e, "Failed to fire alarm");
            }
        }
    }
    if fired > 0 || failed > 0 {
        tracing::debug!(fired, failed, "alarm sweep complete");
    }
    Ok(())
}

/// Cancellable alarm-sweep loop: sleep, then fire due alarms on each tick.
/// The first tick acts as the boot-time overdue scan.
pub async fn run_alarm_sweep_loop() {
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(Duration::from_secs(1)).await {
            break;
        }
        if let Err(e) = run_alarm_sweep(50).await {
            tracing::warn!(error = %e, "alarm sweep failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::ProbeFile;

    #[test]
    fn format_fire_time_renders_local_and_utc() {
        let out = format_fire_time("2026-08-28T07:30:00+00:00").unwrap();
        assert!(
            out.ends_with("local time (2026-08-28 07:30:00 UTC)"),
            "got: {out}"
        );
        assert!(!out.contains("UTC UTC"), "double-UTC in output: {out}");
    }

    #[test]
    fn format_fire_time_rejects_invalid() {
        assert!(format_fire_time("not-a-time").is_err());
    }

    #[test]
    fn next_periodic_fire_skips_missed_periods_in_one_step() {
        // 25s past a 10s-periodic fire: advance to the next whole period
        // (not per-period looping), saturating past long downtime.
        let now = "2026-08-28T00:00:25Z";
        let next_fire = "2026-08-28T00:00:00Z";
        let out = next_periodic_fire(now, next_fire, 10).unwrap();
        assert_eq!(out, "2026-08-28T00:00:30+00:00");
    }

    #[test]
    fn next_periodic_fire_saturates_absurd_interval() {
        // An interval near i64::MAX must never panic the sweep; it saturates to
        // a far-future fire time instead of overflowing a DateTime add.
        let out =
            next_periodic_fire("2026-08-28T00:00:00Z", "2026-08-28T00:00:00Z", i64::MAX).unwrap();
        let parsed = db::parse_utc_timestamp(&out).unwrap();
        assert!(
            parsed > db::parse_utc_timestamp("2026-08-28T00:00:00Z").unwrap(),
            "advance must move past now, got {out}"
        );
    }

    #[tokio::test]
    async fn add_alarm_rejects_past_one_shot() {
        let err = add_alarm(
            "session-a",
            "alice",
            "remind me",
            Some("2020-01-01T00:00:00Z"),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("past"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_rejects_short_periodic_interval() {
        let err = add_alarm("session-a", "alice", "remind me", None, Some(4), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("at least 5 seconds"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_rejects_absurd_periodic_interval() {
        let err = add_alarm(
            "session-a",
            "alice",
            "remind me",
            None,
            Some(u64::MAX),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("at most 292 years"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_requires_exactly_one_of_fire_at_or_interval() {
        // Neither provided.
        let err = add_alarm("session-a", "alice", "remind me", None, None, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Exactly one"), "got: {err}");

        // Both provided.
        let err = add_alarm(
            "session-a",
            "alice",
            "remind me",
            Some("2099-01-01T00:00:00Z"),
            Some(60),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("Exactly one"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_enforces_active_cap() {
        crate::util::test::init_test_stores().await;
        let session = "cap-session";
        for i in 0..10 {
            let fire = format!("2099-01-01T00:00:{i:02}Z");
            add_alarm(
                session,
                "alice",
                &format!("reminder {i}"),
                Some(&fire),
                None,
                None,
            )
            .await
            .unwrap();
        }
        let err = add_alarm(
            session,
            "alice",
            "eleventh",
            Some("2099-01-01T00:01:00Z"),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("limit reached"), "got: {err}");
    }

    /// A trigger's args, as the map `json!` builds for them.
    fn args_of(value: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    /// The stored trigger — the tool's name together with its arguments — is
    /// bounded as a whole, so a short-name/over-long-args trigger and a
    /// long-name/empty-args one are both refused.
    #[tokio::test]
    async fn add_alarm_rejects_overlong_trigger() {
        crate::util::test::init_test_stores().await;
        let overlong = |trigger: Trigger| async move {
            add_alarm(
                "session-a",
                "alice",
                "remind me",
                Some("2099-01-01T00:00:00Z"),
                None,
                Some(trigger),
            )
            .await
            .unwrap_err()
            .to_string()
        };

        let err = overlong(Trigger {
            tool: "weather".to_string(),
            args: args_of(&serde_json::json!({ "city": "x".repeat(MAX_ALARM_TRIGGER_CHARS + 1) })),
        })
        .await;
        assert!(err.contains("too long"), "got: {err}");

        // The name counts too: these arguments alone are far inside the bound.
        let err = overlong(Trigger {
            tool: "x".repeat(MAX_ALARM_TRIGGER_CHARS),
            args: args_of(&serde_json::json!({ "city": "Minsk" })),
        })
        .await;
        assert!(err.contains("too long"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_stores_trigger() {
        crate::util::test::init_test_stores().await;
        let session = "trigger-session";
        let alarm = add_alarm(
            session,
            "alice",
            "remind me",
            Some("2099-01-01T00:00:00Z"),
            None,
            Some(weather_trigger()),
        )
        .await
        .unwrap();
        let trigger = alarm.trigger.as_ref().expect("stored trigger");
        assert_eq!(trigger.render(), "weather {\"city\":\"Minsk\"}");
        let listed = list_alarms(session).await.unwrap();
        assert_eq!(listed.len(), 1, "one triggered alarm must be listed");
        let listed_trigger = listed[0].trigger.as_ref().expect("listed trigger");
        assert_eq!(listed_trigger.render(), "weather {\"city\":\"Minsk\"}");
    }

    fn weather_trigger() -> Trigger {
        let mut args = serde_json::Map::new();
        args.insert("city".to_string(), serde_json::json!("Minsk"));
        Trigger {
            tool: "weather".to_string(),
            args,
        }
    }

    fn trigger_alarm() -> Alarm {
        Alarm {
            id: "alarm-trigger".to_string(),
            session_id: "assistant:alice".to_string(),
            user_name: "alice".to_string(),
            kind: "periodic".to_string(),
            text: "check the thing".to_string(),
            interval_seconds: Some(60),
            next_fire_at: "2026-09-04T12:00:00+00:00".to_string(),
            trigger: Some(StoredTrigger::Tool(weather_trigger())),
        }
    }

    /// A finished run as the wake decision and the notice see it: `output` is
    /// what survives the ANSI strip, `has_output` what the raw streams emitted.
    fn ran(success: bool, detail: &str, output: &str, has_output: bool) -> ProgramOutcome {
        ProgramOutcome {
            success,
            detail: detail.to_string(),
            output: output.to_string(),
            has_output,
        }
    }

    #[test]
    fn trigger_notification_stays_silent_when_check_reports_nothing() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        assert!(
            trigger_notification(&alarm, &trigger, &TriggerOutcome::Silent, None).is_none(),
            "a silent check must not wake the assistant"
        );
    }

    /// The reversal the trigger exists for: only a clean run that reported
    /// nothing at all stays silent — nothing about the content is examined.
    #[test]
    fn wake_decision_reads_only_whether_the_run_reported_anything() {
        let silent = |outcome| matches!(outcome, TriggerOutcome::Silent);
        assert!(
            silent(TriggerOutcome::from_run(ran(
                true,
                "exit status 0",
                "",
                false
            ))),
            "a clean run that said nothing must not wake"
        );
        // Any output wakes — including output that merely resembles a
        // credential, which is reported rather than suppressed.
        assert!(!silent(TriggerOutcome::from_run(ran(
            true,
            "exit status 0",
            "key: sk-live-1234",
            true
        ))));
        // Escape-only output strips to nothing readable but is still output.
        assert!(!silent(TriggerOutcome::from_run(ran(
            true,
            "exit status 0",
            "",
            true
        ))));
        // A failed run wakes even when it said nothing.
        assert!(!silent(TriggerOutcome::from_run(ran(
            false,
            "exit status 2",
            "",
            false
        ))));
    }

    #[test]
    fn trigger_notification_wakes_on_output() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let outcome = TriggerOutcome::Ran(ran(true, "exit status 0", "all good", true));
        let content = trigger_notification(&alarm, &trigger, &outcome, None).unwrap();
        assert!(content.contains("<alarm-notification>"));
        assert!(content.contains("check the thing"));
        assert!(content.contains("weather {\"city\":\"Minsk\"}"));
        assert!(content.contains("exited successfully (exit status 0)"));
        assert!(content.contains("all good"));
        assert!(
            !content.contains("DELETED"),
            "clean path must not report a deletion"
        );
    }

    /// Output that strips away to nothing is still reported, so the notice must
    /// not label it "no output".
    #[test]
    fn trigger_notification_does_not_call_escape_only_output_empty() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let outcome = TriggerOutcome::Ran(ran(true, "exit status 0", "", true));
        let content = trigger_notification(&alarm, &trigger, &outcome, None).unwrap();
        assert!(content.contains("not printable"), "got: {content}");
        assert!(!content.contains("(no output)"), "got: {content}");
    }

    /// A failed check always wakes, empty output included, and the notice
    /// reports whether the alarm was actually removed.
    #[test]
    fn trigger_notification_wakes_on_failure_and_reports_the_deletion_outcome() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let outcome = TriggerOutcome::Ran(ran(false, "exit status 2", "", false));

        let deleted = trigger_notification(&alarm, &trigger, &outcome, Some(&Ok(()))).unwrap();
        assert!(deleted.contains("FAILED (exit status 2)"), "got: {deleted}");
        assert!(deleted.contains("(no output)"), "got: {deleted}");
        assert!(deleted.contains("DELETED"), "got: {deleted}");

        let failed = trigger_notification(
            &alarm,
            &trigger,
            &outcome,
            Some(&Err(anyhow::anyhow!("db down"))),
        )
        .unwrap();
        assert!(failed.contains("FAILED"), "got: {failed}");
        assert!(failed.contains("delete"), "got: {failed}");
        assert!(
            !failed.contains("was DELETED"),
            "must not claim a successful deletion: {failed}"
        );
    }

    #[test]
    fn trigger_notification_reports_a_check_that_could_not_run() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let outcome =
            TriggerOutcome::CouldNotRun("the custom tool \"weather\" is no longer usable".into());
        let content = trigger_notification(&alarm, &trigger, &outcome, Some(&Ok(()))).unwrap();
        assert!(
            content.contains("no longer usable"),
            "the reason must reach the assistant: {content}"
        );
        assert!(content.contains("weather {\"city\":\"Minsk\"}"));
        assert!(content.contains("DELETED"), "got: {content}");
        assert!(
            !content.contains("FAILED"),
            "a check that never ran did not fail: {content}"
        );
    }

    #[test]
    fn in_flight_claim_guards_periodic_overlap() {
        assert!(claim_in_flight("alarm-overlap"));
        // A second claim while the first run is in flight is rejected...
        assert!(!claim_in_flight("alarm-overlap"));
        // ...and released once the run's guard drops.
        drop(InFlightGuard("alarm-overlap".to_string()));
        assert!(claim_in_flight("alarm-overlap"));
        drop(InFlightGuard("alarm-overlap".to_string()));
    }

    /// An alarms row's status, for the fire-time assertions that a non-clean
    /// check took the alarm out of service.
    async fn alarm_status(id: &str) -> String {
        store()
            .conn
            .query("SELECT status FROM alarms WHERE id = ?1", db::params![id])
            .await
            .unwrap()
            .first()
            .map(|r| r.get(0))
            .transpose()
            .unwrap()
            .expect("alarm row must exist")
    }

    /// Arm one one-shot trigger alarm for `owner` under the agent `session_id`
    /// and return it with the receiver its notification will be routed to.
    async fn arm_trigger_alarm(
        session_id: &str,
        owner: &str,
        tool: &str,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> (
        Alarm,
        tokio::sync::mpsc::UnboundedReceiver<crate::agent::message_router::AgentJob>,
    ) {
        seed_owner(owner).await;
        let rx = crate::agent::message_router::register_agent(session_id);
        let alarm = add_alarm(
            session_id,
            owner,
            "poll the thing",
            Some("2099-01-01T00:00:00Z"),
            None,
            Some(Trigger {
                tool: tool.to_string(),
                args,
            }),
        )
        .await
        .expect("arm the trigger alarm");
        (alarm, rx)
    }

    /// The initialized user store with a row for `name`: availability is settled
    /// against it, so every alarm owner needs one.
    async fn seed_owner(name: &str) -> &'static crate::users::UserStore {
        crate::util::test::init_management_test_stores().await;
        let store = crate::users::USER_STORE
            .get()
            .expect("user store initialized");
        store.add_user(name).await.unwrap();
        store
    }

    /// Write `name.ts` into the catalogue folder the tests read (the admin's
    /// `shared/`), removed again when the guard drops.
    fn catalogue_probe(name: &str, header: &str) -> ProbeFile {
        let dir =
            crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME).join("shared");
        std::fs::create_dir_all(&dir).expect("create the shared folder");
        let probe = ProbeFile(dir.join(format!("{name}.ts")));
        std::fs::write(&probe.0, header).expect("write the probe tool");
        probe
    }

    /// Fire `alarm`, wait for the notification the assistant must receive, and
    /// assert the alarm was taken out of service. Returns the notification.
    async fn fire_and_collect_notification(
        alarm: &Alarm,
        session_id: &str,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::agent::message_router::AgentJob>,
    ) -> String {
        fire_alarm(alarm).await.unwrap();
        let job = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("check resolution must finish in time")
            .expect("a check that could not run must notify");
        crate::agent::message_router::unregister_agent(session_id);
        assert!(job.content.contains("DELETED"), "got: {}", job.content);
        assert_eq!(
            alarm_status(&alarm.id).await,
            "removed",
            "a check that cannot run must not re-fire"
        );
        job.content
    }

    /// Fire-time integration: a plain alarm routes its reminder into the
    /// assistant's session and terminalizes the one-shot.
    #[tokio::test]
    async fn fire_plain_alarm_delivers_reminder_and_terminalizes_the_one_shot() {
        crate::util::test::init_management_test_stores().await;
        // A registered receiver captures the routed job deterministically —
        // no consumer loop (and no agent run) is spawned.
        let mut rx = crate::agent::message_router::register_agent("assistant:alarm-plain-it");
        let alarm = add_alarm(
            "assistant:alarm-plain-it",
            "alarm-plain",
            "check the deploy",
            Some("2099-01-01T00:00:00Z"),
            None,
            None,
        )
        .await
        .unwrap();

        fire_alarm(&alarm).await.unwrap();

        let job = rx.recv().await.expect("plain reminder must route");
        assert!(job.content.contains("<alarm-notification>"));
        assert!(job.content.contains("check the deploy"));
        crate::agent::message_router::unregister_agent("assistant:alarm-plain-it");

        assert_eq!(
            alarm_status(&alarm.id).await,
            "fired",
            "one-shot must be terminalized"
        );
    }

    /// Fire-time integration: a trigger naming a tool the catalogue does not
    /// have cannot run, so the assistant is told why and the alarm is removed.
    #[tokio::test]
    async fn fire_trigger_alarm_reports_a_tool_that_is_gone_and_removes_the_alarm() {
        let (alarm, mut rx) = arm_trigger_alarm(
            "assistant:alarm-gone-it",
            crate::users::ADMIN_USER_NAME,
            "no_such_alarm_tool",
            serde_json::Map::new(),
        )
        .await;

        let content =
            fire_and_collect_notification(&alarm, "assistant:alarm-gone-it", &mut rx).await;
        assert!(content.contains("<alarm-notification>"));
        assert!(content.contains("poll the thing"));
        assert!(content.contains("no_such_alarm_tool"), "got: {content}");
        assert!(content.contains("no longer usable"), "got: {content}");
    }

    /// Fire-time integration: a grant taken away between arming and firing is a
    /// check that cannot run — its own wording, and the alarm goes with it.
    #[tokio::test]
    async fn fire_trigger_alarm_reports_a_revoked_grant_and_removes_the_alarm() {
        let owner = "revoked_grant_guest";
        let store = seed_owner(owner).await;
        store.add_grant(owner, "alarm_grant_probe").await.unwrap();
        let (alarm, mut rx) = arm_trigger_alarm(
            "assistant:alarm-revoked-it",
            owner,
            "alarm_grant_probe",
            serde_json::Map::new(),
        )
        .await;
        // The tool itself stays fine — it is only the grant that goes.
        let _probe = catalogue_probe("alarm_grant_probe", "// @description Probe.\n");
        store
            .remove_grant(owner, "alarm_grant_probe")
            .await
            .unwrap();

        let content =
            fire_and_collect_notification(&alarm, "assistant:alarm-revoked-it", &mut rx).await;
        assert!(
            content.contains("no longer available to you"),
            "got: {content}"
        );
    }

    /// Fire-time integration: a stored trigger whose arguments no longer fit
    /// the tool's rewritten header cannot run, so the assistant is told why and
    /// the alarm is removed.
    #[tokio::test]
    async fn fire_trigger_alarm_reports_arguments_that_no_longer_fit() {
        let (alarm, mut rx) = arm_trigger_alarm(
            "assistant:alarm-args-it",
            crate::users::ADMIN_USER_NAME,
            "alarm_args_probe",
            args_of(&serde_json::json!({ "city": "Minsk" })),
        )
        .await;
        // The tool's header no longer declares the parameter the trigger stored.
        let _probe = catalogue_probe("alarm_args_probe", "// @description Probe.\n");

        let content =
            fire_and_collect_notification(&alarm, "assistant:alarm-args-it", &mut rx).await;
        assert!(content.contains("no longer fit"), "got: {content}");
        assert!(
            content.contains("[ignored arguments: city]"),
            "got: {content}"
        );
    }

    /// Fire-time integration: a stored check that cannot be read is a check that
    /// cannot run — the assistant is told why and the alarm goes with it.
    #[tokio::test]
    async fn fire_trigger_alarm_reports_a_stored_check_it_cannot_read() {
        let session = "assistant:alarm-unreadable-it";
        let (alarm, mut rx) = arm_trigger_alarm(
            session,
            crate::users::ADMIN_USER_NAME,
            "alarm_unreadable_probe",
            serde_json::Map::new(),
        )
        .await;
        // Replace the stored check with a value that is not one, and make the
        // alarm due so the sweep's own read path is what fires it.
        store()
            .conn
            .execute(
                "UPDATE alarms SET trigger = ?1, next_fire_at = ?2 WHERE id = ?3",
                db::params![
                    "not a stored check",
                    "2020-01-01T00:00:00Z",
                    alarm.id.as_str()
                ],
            )
            .await
            .unwrap();
        let due = due_alarms(&db::now()).await.unwrap();
        let stored = due
            .iter()
            .find(|candidate| candidate.id == alarm.id)
            .expect("the alarm must be due");

        let content = fire_and_collect_notification(stored, session, &mut rx).await;
        assert!(content.contains("cannot be read"), "got: {content}");
        assert!(
            content.contains("unreadable stored check"),
            "the notice must show a marker where the check would be: {content}"
        );
    }
}
