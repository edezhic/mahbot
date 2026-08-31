//! Alarm/reminder persistence and fire routing for the Assistant session.
//!
//! Backs the `add_alarm` / `list_alarms` / `remove_alarm` tools and the
//! periodic background sweep that routes due reminders back into the
//! Assistant's own personal session as user-role messages.

use std::time::Duration;

use anyhow::Context as _;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration};

use crate::Role;
use crate::agent::message_router::{AgentJob, MessageKind};
use crate::db;

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
        interval_seconds: row.get(COL_ALARM_INTERVAL_SECONDS)?,
        next_fire_at: row.get(COL_ALARM_NEXT_FIRE_AT)?,
    })
}

/// Maximum number of active alarms allowed per session.
const MAX_ACTIVE_ALARMS: i64 = 10;

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
) -> Result<Alarm> {
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
             (id, session_id, user_name, kind, text, fire_at, interval_seconds, next_fire_at, status, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'active', ?9)",
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
            ],
        )
        .await?;

    Ok(Alarm {
        id,
        session_id: session_id.to_string(),
        user_name: user_name.to_string(),
        kind: kind.to_string(),
        text: text.to_string(),
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
/// The notification is delivered as a user-role message into the calling
/// assistant's own session (`alarm.session_id`, resolved directly — never
/// re-derived from the id, which would double-escape colliding user names).
/// One-shot alarms are terminalized (`status='fired'`); periodic alarms advance
/// `next_fire_at` past every missed whole period.
pub(crate) async fn fire_alarm(alarm: &Alarm) -> Result<()> {
    let now = db::now();

    // Delivery is sourced from the stored raw user/workspace so a reminder
    // targets the right personal session regardless of agent-ID escaping.
    let user = &alarm.user_name;
    let workspace_name = format!("personal:{user}");
    let agent_id = alarm.session_id.clone();
    let content = format!(
        "<alarm-notification>\nYour reminder \"{}\" is due now (scheduled for {}).\n</alarm-notification>",
        alarm.text, alarm.next_fire_at
    );
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
    // Persist a durable envelope BEFORE advancing state so a crash between the
    // state advance and the consumer delivering the message replays the
    // reminder at boot — closes the loss side of at-least-once. A persistence
    // failure degrades to best-effort (at-most-once) routing.
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
        let next = next_periodic_fire(&now, &alarm.next_fire_at, interval)?;
        store()
            .conn
            .execute(
                "UPDATE alarms SET next_fire_at = ?1 WHERE id = ?2",
                db::params![next, alarm.id.as_str()],
            )
            .await?;
    }

    // Route the (now durable) notification into the Assistant session.
    crate::agent::message_router::route(&agent_id, job);
    Ok(())
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
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("past"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_rejects_short_periodic_interval() {
        let err = add_alarm("session-a", "alice", "remind me", None, Some(5))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("at least 10 seconds"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn add_alarm_rejects_absurd_periodic_interval() {
        let err = add_alarm("session-a", "alice", "remind me", None, Some(u64::MAX))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("at most 292 years"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_requires_exactly_one_of_fire_at_or_interval() {
        // Neither provided.
        let err = add_alarm("session-a", "alice", "remind me", None, None)
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
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("limit reached"), "got: {err}");
    }
}
