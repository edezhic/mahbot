//! Alarm/reminder persistence and fire routing for the Assistant session.
//!
//! Backs the `add_alarm` / `list_alarms` / `remove_alarm` tools and the
//! periodic background sweep that routes due reminders back into the
//! Assistant's own personal session as user-role messages.

use std::collections::HashSet;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Context as _;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration};

use crate::Role;
use crate::agent::message_router::{AgentJob, MessageKind};
use crate::db;
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
        COMMAND          => "command",
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
    /// Optional shell command to run at fire time (command-armed alarm).
    pub command: Option<String>,
    /// Periodic interval in seconds.
    pub interval_seconds: Option<i64>,
    /// RFC3339 UTC next fire time.
    pub next_fire_at: String,
}

fn alarm_from_row(row: &db::Row) -> Result<Alarm, ::turso::Error> {
    Ok(Alarm {
        id: row.get(COL_ALARM_ID)?,
        session_id: row.get(COL_ALARM_SESSION_ID)?,
        user_name: row.get(COL_ALARM_USER_NAME)?,
        kind: row.get(COL_ALARM_KIND)?,
        text: row.get(COL_ALARM_TEXT)?,
        command: row.get(COL_ALARM_COMMAND)?,
        interval_seconds: row.get(COL_ALARM_INTERVAL_SECONDS)?,
        next_fire_at: row.get(COL_ALARM_NEXT_FIRE_AT)?,
    })
}

/// Maximum number of active alarms allowed per session.
const MAX_ACTIVE_ALARMS: i64 = 10;

/// Upper bound on a command-armed alarm's shell command length. Keeps a
/// maliciously-or-over-eagerly-long command from blowing the notification
/// envelope / task spawn budget.
const MAX_ALARM_COMMAND_CHARS: usize = 2000;

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
/// the future; periodic intervals must be at least 10 seconds. At most
/// [`MAX_ACTIVE_ALARMS`] active alarms may exist per session.
pub(crate) async fn add_alarm(
    session_id: &str,
    user_name: &str,
    text: &str,
    fire_at: Option<&str>,
    interval_seconds: Option<u64>,
    command: Option<&str>,
) -> Result<Alarm> {
    // Validate an optional command-arming payload before touching the DB.
    if let Some(cmd) = command {
        anyhow::ensure!(!cmd.trim().is_empty(), "Alarm command must not be empty");
        anyhow::ensure!(
            cmd.chars().count() <= MAX_ALARM_COMMAND_CHARS,
            "Alarm command too long (maximum {MAX_ALARM_COMMAND_CHARS} characters)"
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
            anyhow::ensure!(interval >= 10, "Period must be at least 10 seconds");
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
             (id, session_id, user_name, kind, text, fire_at, interval_seconds, next_fire_at, status, created_at, command) \
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
                command,
            ],
        )
        .await?;

    Ok(Alarm {
        id,
        session_id: session_id.to_string(),
        user_name: user_name.to_string(),
        kind: kind.to_string(),
        text: text.to_string(),
        command: command.map(str::to_string),
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

/// The due alarms at `now` (RFC3339 UTC), ordered by next fire time.
pub(crate) async fn due_alarms(now: &str) -> Result<Vec<Alarm>> {
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
/// Command-armed alarms (`command` set) re-verify ownership, run the command
/// in the owner's personal workspace, and deliver the command output; plain
/// alarms deliver the reminder text. The notification is delivered as a
/// user-role message into the calling assistant's own session (`alarm.session_id`,
/// resolved directly — never re-derived from the id, which would double-escape
/// colliding user names). One-shot alarms are terminalized (`status='fired'`);
/// periodic alarms advance `next_fire_at` past every missed whole period.
pub(crate) async fn fire_alarm(alarm: &Alarm) -> Result<()> {
    match &alarm.command {
        Some(command) => fire_command_alarm(alarm, command).await,
        None => fire_plain_alarm(alarm).await,
    }
}

/// Deliver a plain (non-command) alarm reminder and advance its state.
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
    deliver_alarm_notification(alarm, content).await?;
    advance_alarm_state(alarm, &now).await?;
    Ok(())
}

/// Build the durable [`AgentJob`] envelope for an alarm delivery, persist it
/// best-effort (a persistence failure degrades to at-most-once routing), and
/// route it into the calling assistant's session.
async fn deliver_alarm_notification(alarm: &Alarm, content: String) -> Result<()> {
    // Scrub the rendered content BEFORE it is persisted: alarm text, commands,
    // and command output are user/model-supplied and can embed credentials.
    let content = crate::util::scrub_credentials(&content);
    // Delivery is sourced from the stored raw user/workspace so a reminder
    // targets the right personal session regardless of agent-ID escaping.
    let user = &alarm.user_name;
    let workspace_name = format!("personal:{user}");
    let agent_id = alarm.session_id.clone();
    let mut job = AgentJob {
        content,
        workspace_name,
        user_name: user.clone(),
        // History-attribution tag only — the reply is broadcast to all of the
        // user's channel bindings, so no single transport is claimed here.
        channel: "gui".to_string(),
        kind: MessageKind::UserMessage,
        role: Role::Assistant,
        reply_target: None,
        pending_job_id: None,
    };
    // Persist a durable envelope BEFORE routing so a crash after persisting
    // but before the consumer delivers the message replays the reminder at
    // boot — closes the loss side of at-least-once. A persistence failure
    // degrades to best-effort (at-most-once) routing.
    let id = crate::generate_id();
    let persisted = match crate::agent::message_router::persist_pending(&job, id.clone()).await {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(alarm = %alarm.id, error = %e, "Failed to persist alarm delivery — routing best-effort");
            false
        }
    };
    if persisted {
        job.pending_job_id = Some(id);
    }

    // Route the (now durable) notification into the Assistant session.
    crate::agent::message_router::route(&agent_id, job);
    Ok(())
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

/// Alarms whose command run is currently in flight, keyed by alarm id — the
/// periodic-overlap guard that prevents a slow command from piling up one
/// spawn per sweep.
static COMMANDS_IN_FLIGHT: LazyLock<std::sync::Mutex<HashSet<String>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

/// Fire a command-armed alarm: re-verify ownership, advance state, then spawn
/// a detached task that runs the command and delivers its output.
async fn fire_command_alarm(alarm: &Alarm, command: &str) -> Result<()> {
    // Admin re-verification at fire time: the owner must still have admin
    // rights, else degrade to a plain reminder.
    if !crate::users::is_admin(&alarm.user_name).await {
        return fire_plain_alarm(alarm).await;
    }

    // Advance state FIRST — independent of the command outcome — so a slow or
    // failing command cannot keep the alarm due and re-fire it on every sweep.
    advance_alarm_state(alarm, &db::now()).await?;

    // Periodic-overlap guard: if a previous run of this alarm is still in
    // flight, skip this firing entirely (advanced but no spawn, no notification).
    if !claim_in_flight(&alarm.id) {
        return Ok(());
    }

    tokio::spawn(run_alarm_command_task(alarm.clone(), command.to_string()));
    Ok(())
}

/// Try to mark an alarm id's command run as in flight: `true` claims it (the
/// spawned task's [`InFlightGuard`] releases it), `false` means a previous run
/// of the same alarm's command is still active.
fn claim_in_flight(alarm_id: &str) -> bool {
    COMMANDS_IN_FLIGHT
        .lock()
        .unwrap_poison()
        .insert(alarm_id.to_string())
}

/// RAII guard that removes an alarm id from [`COMMANDS_IN_FLIGHT`] on drop, so
/// every exit path (including a panic/abort of the spawned command task) clears
/// the in-flight marker and lets the next firing run.
struct InFlightGuard(String);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        COMMANDS_IN_FLIGHT.lock().unwrap_poison().remove(&self.0);
    }
}

/// Run a command-armed alarm's command in the owner's personal workspace and
/// deliver its notification — or stay silent when the command succeeded with
/// no output (see [`alarm_command_notification`]).
async fn run_alarm_command_task(alarm: Alarm, command: String) {
    // The in-flight marker is removed unconditionally on drop.
    let _in_flight = InFlightGuard(alarm.id.clone());

    let ws = crate::users::personal_workspace_struct(&alarm.user_name);
    let outcome = crate::tools::shell::run_raw_command(&ws, &command).await;

    match alarm_command_notification(&alarm, &command, &outcome) {
        Some(content) => {
            if let Err(e) = deliver_alarm_notification(&alarm, content).await {
                tracing::warn!(alarm = %alarm.id, error = %e, "Failed to deliver alarm command notification");
            }
        }
        None => {
            tracing::info!(alarm = %alarm.id, "alarm command produced no output — staying silent");
        }
    }
}

/// Decide the delivery for a finished alarm-command run: `None` stays silent
/// (the command succeeded with no output); `Some` is the rendered
/// `<alarm-notification>` content with the status line and the (scrubbed,
/// truncated) command output.
///
/// The wake/no-wake signal is `has_output`, computed pre-redaction — so
/// ANSI-only or fully credential-redacted output counts as empty.
fn alarm_command_notification(
    alarm: &Alarm,
    command: &str,
    outcome: &crate::tools::shell::RawCommandOutcome,
) -> Option<String> {
    if outcome.success && !outcome.has_output {
        return None;
    }
    let output_display = if outcome.has_output {
        crate::util::truncate_sandwich(
            &outcome.output,
            crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "alarm command output",
        )
    } else {
        "(no output)".to_string()
    };
    let status_line = if outcome.success {
        format!("The command exited successfully ({}).", outcome.detail)
    } else {
        format!("The command FAILED ({}).", outcome.detail)
    };
    Some(crate::prompt::substitute(
        &crate::prompt::load_prompt("alarm_command_notification.md"),
        &[
            ("{{text}}", &alarm.text),
            ("{{fire_at}}", &alarm.next_fire_at),
            ("{{command}}", command),
            ("{{command_status}}", &status_line),
            ("{{command_output}}", &output_display),
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
pub(crate) async fn run_alarm_sweep(batch_limit: usize) -> Result<()> {
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
        tracing::info!(fired, failed, "alarm sweep complete");
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
        let err = add_alarm("session-a", "alice", "remind me", None, Some(5), None)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("at least 10 seconds"),
            "got: {err}"
        );
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

    #[tokio::test]
    async fn add_alarm_rejects_blank_command() {
        let err = add_alarm(
            "session-a",
            "alice",
            "remind me",
            Some("2099-01-01T00:00:00Z"),
            None,
            Some("   "),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("must not be empty"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_rejects_overlong_command() {
        let long = "x".repeat(MAX_ALARM_COMMAND_CHARS + 1);
        let err = add_alarm(
            "session-a",
            "alice",
            "remind me",
            Some("2099-01-01T00:00:00Z"),
            None,
            Some(long.as_str()),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("too long"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_stores_command() {
        crate::util::test::init_test_stores().await;
        let session = "cmd-session";
        let alarm = add_alarm(
            session,
            "alice",
            "remind me",
            Some("2099-01-01T00:00:00Z"),
            None,
            Some("echo hi"),
        )
        .await
        .unwrap();
        assert_eq!(alarm.command.as_deref(), Some("echo hi"));
        let listed = list_alarms(session).await.unwrap();
        assert_eq!(listed.len(), 1, "one command-armed alarm must be listed");
        assert_eq!(listed[0].command.as_deref(), Some("echo hi"));
    }

    fn command_alarm() -> Alarm {
        Alarm {
            id: "alarm-cmd".to_string(),
            session_id: "assistant:alice".to_string(),
            user_name: "alice".to_string(),
            kind: "periodic".to_string(),
            text: "check the thing".to_string(),
            interval_seconds: Some(60),
            next_fire_at: "2026-09-04T12:00:00+00:00".to_string(),
            command: Some("echo hi".to_string()),
        }
    }

    fn raw_outcome(
        success: bool,
        has_output: bool,
        output: &str,
    ) -> crate::tools::shell::RawCommandOutcome {
        crate::tools::shell::RawCommandOutcome {
            success,
            detail: if success {
                "exit status 0".to_string()
            } else {
                "exit status 2".to_string()
            },
            has_output,
            output: output.to_string(),
        }
    }
    #[test]
    fn alarm_command_notification_stays_silent_on_clean_success() {
        let alarm = command_alarm();
        let outcome = raw_outcome(true, false, "");
        assert!(alarm_command_notification(&alarm, "echo hi", &outcome).is_none());
    }

    #[test]
    fn alarm_command_notification_wakes_on_output() {
        let alarm = command_alarm();
        let outcome = raw_outcome(true, true, "all good");
        let content = alarm_command_notification(&alarm, "echo hi", &outcome).unwrap();
        assert!(content.contains("<alarm-notification>"));
        assert!(content.contains("check the thing"));
        assert!(content.contains("`echo hi`"));
        assert!(content.contains("exited successfully (exit status 0)"));
        assert!(content.contains("all good"));
    }

    #[test]
    fn alarm_command_notification_wakes_on_failure_even_when_empty() {
        let alarm = command_alarm();
        let outcome = raw_outcome(false, false, "");
        let content = alarm_command_notification(&alarm, "echo hi", &outcome).unwrap();
        assert!(content.contains("FAILED (exit status 2)"), "got: {content}");
        assert!(content.contains("(no output)"), "got: {content}");
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

    /// Fire-time integration: a command-armed alarm whose owner is NOT an
    /// admin degrades to a plain reminder — the routed notification carries
    /// only the reminder text (no command section, no command execution) and
    /// the one-shot is terminalized.
    #[tokio::test]
    async fn fire_alarm_degrades_command_to_plain_for_non_admin_owner() {
        crate::util::test::init_test_stores().await;
        let _ = crate::agent::message_router::init_global();
        // A registered receiver captures the routed job deterministically —
        // no consumer loop (and no agent run) is spawned.
        let mut rx = crate::agent::message_router::register_agent("assistant:alarm-bob");
        // "alarm-bob" has no user row → not an admin at fire time.
        let mut alarm = add_alarm(
            "assistant:alarm-bob",
            "alarm-bob",
            "check the deploy",
            Some("2099-01-01T00:00:00Z"),
            None,
            Some("echo secret-run"),
        )
        .await
        .unwrap();
        // Force the alarm due.
        store()
            .conn
            .execute(
                "UPDATE alarms SET next_fire_at = '2020-01-01T00:00:00+00:00' WHERE id = ?1",
                db::params![alarm.id.as_str()],
            )
            .await
            .unwrap();
        alarm.next_fire_at = "2020-01-01T00:00:00+00:00".to_string();

        fire_alarm(&alarm).await.unwrap();

        let job = rx.recv().await.expect("degraded plain reminder must route");
        assert!(job.content.contains("<alarm-notification>"));
        assert!(job.content.contains("check the deploy"));
        assert!(!job.content.contains("Command:"), "got: {}", job.content);
        assert!(!job.content.contains("secret-run"), "got: {}", job.content);
        crate::agent::message_router::unregister_agent("assistant:alarm-bob");

        let status: String = store()
            .conn
            .query(
                "SELECT status FROM alarms WHERE id = ?1",
                db::params![alarm.id.as_str()],
            )
            .await
            .unwrap()
            .first()
            .map(|r| r.get(0))
            .transpose()
            .unwrap()
            .expect("alarm row must exist");
        assert_eq!(status, "fired", "one-shot must be terminalized");
    }

    /// Fire-time integration: an admin owner's command-armed alarm executes
    /// the command through the raw shell runner and delivers the command
    /// output inside the <alarm-notification>.
    #[tokio::test]
    async fn fire_alarm_executes_command_and_delivers_output_for_admin_owner() {
        crate::util::test::init_test_stores().await;
        let _ = crate::agent::message_router::init_global();
        let owner = "alarm-admin-it";
        crate::users::USER_STORE
            .get()
            .expect("user store initialized")
            .add_user(owner, Some("full"), crate::Role::Assistant)
            .await
            .unwrap();
        let mut rx = crate::agent::message_router::register_agent("assistant:alarm-admin-it");
        let alarm = Alarm {
            id: "alarm-admin-run".to_string(),
            session_id: "assistant:alarm-admin-it".to_string(),
            user_name: owner.to_string(),
            kind: "one-shot".to_string(),
            text: "poll the thing".to_string(),
            interval_seconds: None,
            next_fire_at: "2026-09-04T12:00:00+00:00".to_string(),
            command: Some("echo hello-alarm-run".to_string()),
        };

        fire_alarm(&alarm).await.unwrap();

        let job = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("command run must finish in time")
            .expect("admin command notification must route");
        assert!(job.content.contains("<alarm-notification>"));
        assert!(job.content.contains("poll the thing"));
        assert!(
            job.content.contains("`echo hello-alarm-run`"),
            "got: {}",
            job.content
        );
        assert!(
            job.content.contains("exited successfully"),
            "got: {}",
            job.content
        );
        assert!(
            job.content.contains("hello-alarm-run"),
            "got: {}",
            job.content
        );
        crate::agent::message_router::unregister_agent("assistant:alarm-admin-it");
    }
}
