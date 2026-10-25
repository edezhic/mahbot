//! Alarm/reminder persistence and fire routing for the Assistant session.
//!
//! Backs the `add_alarm` / `list_alarms` / `remove_alarm` tools and the
//! periodic background sweep that routes due reminders back into the
//! Assistant's own personal session as user-role messages.
//!
//! An alarm optionally carries a [`StoredTrigger`] — one custom tool plus its
//! arguments, stored in the `trigger` column. At fire time the tool runs as the
//! alarm's owner in the owner's personal workspace, and only what the check
//! reported decides whether the Assistant is woken at all.
//!
//! A check stopped by its time limit is not a verdict about the alarm: nothing
//! is delivered for that attempt, the occurrence stays pending and is retried
//! after a short cool-down. Every other non-clean outcome — a non-zero exit, the
//! memory ceiling, an output drain overrun, a check that could not run at all —
//! is a broken check and takes the alarm with it. A stored check that cannot be
//! read is a check that cannot run, and is treated as one.

use std::collections::HashSet;
use std::fmt::Write as _;
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
        ID                  => "id",
        SESSION_ID          => "session_id",
        USER_NAME           => "user_name",
        KIND                => "kind",
        TEXT                => "text",
        TRIGGER             => "trigger",
        ROLE                => "role",
        IDENTIFIER          => "identifier",
        WORKSPACE           => "workspace",
        INTERVAL_SECONDS    => "interval_seconds",
        NEXT_FIRE_AT        => "next_fire_at",
        UNFINISHED_ATTEMPTS => "unfinished_attempts",
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
    /// The message the alarm fires or launches with — the card's description.
    pub text: String,
    /// Optional custom-tool check to run at fire time (triggered alarm).
    pub trigger: Option<StoredTrigger>,
    /// The custom role this alarm wakes; `None` wakes the Assistant.
    pub role: Option<String>,
    /// The custom role branch identifier, when one was given.
    pub identifier: Option<String>,
    /// The folder the role's launch runs in (a managed workspace name, a
    /// `personal:{user}` key, or an absolute path); `None` = the owner's
    /// personal folder.
    pub workspace: Option<String>,
    /// Periodic interval in seconds.
    pub interval_seconds: Option<i64>,
    /// RFC3339 UTC next fire time: the moment the pending occurrence is due for.
    /// A retry of an attempt that did not finish in time never moves it.
    pub next_fire_at: String,
    /// Attempts in a row that did not finish in time since the last completed
    /// attempt; drives the one-time "not producing a result" notice and the
    /// LATE note of the report that finally arrives.
    pub unfinished_attempts: i64,
}

impl Alarm {
    /// The alarm's custom-role target as the listing surfaces show it —
    /// `addressed to custom role "analyst" (branch "daily") in my-workspace`,
    /// with the branch and folder only when recorded. `None` for an alarm that
    /// wakes the Assistant.
    pub(crate) fn role_target(&self) -> Option<String> {
        let role = self.role.as_deref()?;
        let mut out = format!("addressed to custom role \"{role}\"");
        if let Some(identifier) = &self.identifier {
            let _ = write!(out, " (branch \"{identifier}\")");
        }
        if let Some(workspace) = &self.workspace {
            let _ = write!(out, " in {workspace}");
        }
        Some(out)
    }
}

/// The custom role an alarm is addressed to (an alarm with no target wakes the
/// Assistant, as before).
pub(crate) struct AlarmTarget {
    pub role: String,
    pub identifier: Option<String>,
    pub workspace: Option<String>,
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

    /// The arguments as they are shown, `None` when the tool takes none.
    fn args_json(&self) -> Option<String> {
        (!self.args.is_empty()).then(|| {
            serde_json::to_string(&self.args)
                .expect("a trigger's arguments are always serializable")
        })
    }

    /// The trigger as it is shown: the tool's name followed by its arguments,
    /// nothing but the name when it has none.
    fn render(&self) -> String {
        match self.args_json() {
            Some(args) => format!("{} {args}", self.tool),
            None => self.tool.clone(),
        }
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

    /// The check's tool name alone — the alarm card's headline — or the marker
    /// when there is nothing readable to name.
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Tool(trigger) => &trigger.tool,
            Self::Unreadable => UNREADABLE_TRIGGER,
        }
    }

    /// The arguments shown beside the name, `None` when the check has none.
    pub(crate) fn args_json(&self) -> Option<String> {
        match self {
            Self::Tool(trigger) => trigger.args_json(),
            Self::Unreadable => None,
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
        role: row.get(COL_ALARM_ROLE)?,
        identifier: row.get(COL_ALARM_IDENTIFIER)?,
        workspace: row.get(COL_ALARM_WORKSPACE)?,
        interval_seconds: row.get(COL_ALARM_INTERVAL_SECONDS)?,
        next_fire_at: row.get(COL_ALARM_NEXT_FIRE_AT)?,
        unfinished_attempts: row.get(COL_ALARM_UNFINISHED_ATTEMPTS)?,
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
const MAX_ACTIVE_ALARMS: i64 = 50;

/// Upper bound on an alarm's whole stored trigger — the tool's name together
/// with its arguments — in characters. Keeps an over-eagerly-large trigger from
/// bloating the notification envelope or the `<user-alarms>` prompt summary.
const MAX_ALARM_TRIGGER_CHARS: usize = 2000;

/// Upper bound on a periodic interval (~292 years): keeps the total swept
/// advance (interval × skipped whole periods) within a representable
/// [`DateTime`] range, so the O(1) periodic skip saturates instead of
/// overflowing. Also rejects absurd intervals as caller errors at creation.
const MAX_PERIOD_SECS: i64 = i64::MAX / 1_000_000_000;

/// The alarm sweep's tick: how long the loop waits between passes over due
/// alarms. The first tick after start is also the overdue scan.
const SWEEP_TICK: Duration = Duration::from_secs(1);

/// A firing later than its scheduled time by more than this is MISSED, and its
/// notice says so: 30 sweep ticks. One tick can overrun — its due batch is fired
/// one alarm at a time, each with a durable write — so a firing past the bound
/// did not come from a routine tick.
#[expect(clippy::cast_possible_wrap)] // 30 ticks fit i64 by many orders
const MISSED_FIRING_TOLERANCE: ChronoDuration =
    ChronoDuration::seconds(30 * SWEEP_TICK.as_secs() as i64);

/// Cool-down before an occurrence whose attempt did not finish in time is tried
/// again: short enough that a starved machine gets the check back quickly, long
/// enough that retries cannot pile up into a burst.
const RETRY_COOLDOWN_SECS: i64 = 60;

/// Consecutive attempts that did not finish in time before the assistant is told
/// — once — that the check is not producing a result: about half an hour with
/// the attempt ceiling and the cool-down.
const UNFINISHED_NOTICE_ATTEMPTS: i64 = 3;

/// Whether a firing at `now` was MISSED — the alarm's scheduled time had
/// already passed by more than [`MISSED_FIRING_TOLERANCE`] when it fired.
/// `scheduled` is the time the firing belongs to: the value as it stood before
/// the state advance, never a periodic alarm's next occurrence.
///
/// Fail-soft: a scheduled time that cannot be read as a time is not a missed
/// firing, and neither is one still in the future — a firing error would leave
/// a due alarm re-firing on every tick.
fn is_missed_firing(now: &str, scheduled: &str) -> bool {
    let (Ok(now), Ok(scheduled)) = (
        db::parse_utc_timestamp(now),
        db::parse_utc_timestamp(scheduled),
    ) else {
        return false;
    };
    now.signed_duration_since(scheduled) > MISSED_FIRING_TOLERANCE
}

/// How the firing a notice belongs to was timed, as the notice says it.
enum Timing {
    /// On time: the occurrence fired when it was scheduled to.
    Due,
    /// Fired long after the alarm was due — the product was not running then.
    Missed,
    /// Earlier attempts at this occurrence did not finish and the report that
    /// finally arrived is late for it.
    Late,
}

/// The opening line of the notices that carry the reminder's own text — the
/// plain reminder and the triggered check's: what the reminder is, and how the
/// firing it belongs to was timed. Exactly one timing statement, so a report
/// that arrives after retries is announced as late instead of stacking a second
/// lateness note on top of the missed-firing one. Every timing names the
/// occurrence's own due moment, which no retry moves.
fn reminder_line(alarm: &Alarm, timing: &Timing) -> String {
    let asset = match timing {
        Timing::Due => "alarm_reminder_due.md",
        Timing::Missed => "alarm_reminder_missed.md",
        Timing::Late => "alarm_reminder_late.md",
    };
    crate::prompt::substitute(
        &crate::prompt::load_prompt(asset),
        &[
            ("{{text}}", &alarm.text),
            ("{{fire_at}}", &alarm.next_fire_at),
        ],
    )
}

/// The timing a notice states. Only a report can be late: a report that settles
/// an occurrence whose earlier attempts did not finish says so instead of the
/// missed-firing note, and a broken check's notice reads as it always has.
fn trigger_timing(alarm: &Alarm, missed: bool, reported: bool) -> Timing {
    if reported && alarm.unfinished_attempts > 0 {
        Timing::Late
    } else if missed {
        Timing::Missed
    } else {
        Timing::Due
    }
}

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

/// Validate an alarm's optional custom-role target before it is stored: only the
/// administrator may address an alarm to a custom role, the role must exist, the
/// branch identifier and folder must resolve to what will be stored, and a role-
/// addressed alarm's text is the launch message so it must not be empty and
/// cannot also carry a check. Returns the role, branch identifier and folder
/// address to store.
async fn validate_alarm_target(
    user_name: &str,
    text: &str,
    target: Option<AlarmTarget>,
    has_trigger: bool,
) -> Result<(Option<String>, Option<String>, Option<String>)> {
    let Some(target) = target else {
        return Ok((None, None, None));
    };
    anyhow::ensure!(
        crate::users::is_admin_name(user_name),
        "usage: only the administrator may address an alarm to a custom role"
    );
    anyhow::ensure!(
        crate::custom_role::get(&target.role).await?.is_some(),
        "no custom role named '{}'",
        target.role
    );
    // An explicit empty/whitespace identifier counts as absent.
    let identifier = target
        .identifier
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty());
    if let Some(id) = &identifier {
        crate::custom_role::validate_identifier(id)?;
    }
    // Same for the workspace, resolved to the address actually stored.
    let workspace = match target
        .workspace
        .map(|w| w.trim().to_string())
        .filter(|w| !w.is_empty())
    {
        None => None,
        Some(raw) => Some(
            crate::launch::launch_workspace(Some(raw.as_str()))
                .await?
                .name,
        ),
    };
    anyhow::ensure!(
        !has_trigger,
        "usage: a role-addressed alarm's text IS the role's launch message, so its \
         check cannot be combined with a role target"
    );
    anyhow::ensure!(
        !text.trim().is_empty(),
        "usage: a role-addressed alarm's text is the role's launch message, so it must not be empty"
    );
    Ok((Some(target.role), identifier, workspace))
}

/// The session an alarm call actually acts on: a custom-role caller's alarms
/// live on the owner's Assistant session, the caller's own id otherwise.
///
/// A custom role runs at the owner's strength, and a plain alarm it arms is
/// still an owner reminder delivered to the assistant — so `add`/`list`/`remove`
/// all resolve the owner's assistant session for such a caller, never the role's
/// own branch (which would hold an alarm it could never see or cancel).
#[must_use]
pub(crate) fn owner_alarm_session(agent_id: &str, user_name: &str) -> String {
    if crate::session::is_custom_role_agent_id(agent_id) {
        crate::session::assistant_agent_id(user_name)
    } else {
        agent_id.to_string()
    }
}

/// Create a new alarm for a session.
///
/// Exactly one of `fire_at` (one-shot) or `interval_seconds` (periodic) must be
/// provided. One-shot fire times are normalized to RFC3339 UTC and must be in
/// the future; periodic intervals must be at least 5 seconds. At most
/// [`MAX_ACTIVE_ALARMS`] active alarms may exist per session.
///
/// `target` addresses the alarm to a custom role instead of the Assistant: only
/// the administrator may do that, the role must exist, and the alarm carries no
/// trigger — its text IS the role's launch message.
pub(crate) async fn add_alarm(
    session_id: &str,
    user_name: &str,
    text: &str,
    fire_at: Option<&str>,
    interval_seconds: Option<u64>,
    trigger: Option<Trigger>,
    target: Option<AlarmTarget>,
) -> Result<Alarm> {
    // A role-addressed alarm the owner's own Assistant never sees belongs on
    // the assistant's session: a custom role may hold the alarm tools, but a
    // plain alarm it arms is still an owner reminder.
    let session_id = owner_alarm_session(session_id, user_name);

    // Validate the target before touching the DB: a broken target must never be
    // armed. A role-addressed alarm's text is the launch message, so a trigger
    // can never ride along and the text must be non-empty.
    let (role, identifier, workspace) =
        validate_alarm_target(user_name, text, target, trigger.is_some()).await?;

    // Validate an optional trigger before touching the DB: the whole stored
    // trigger — the tool's name together with its arguments — is bounded.
    let encoded_trigger = trigger.as_ref().map(Trigger::encode);
    if let Some(encoded) = &encoded_trigger {
        anyhow::ensure!(
            encoded.chars().count() <= MAX_ALARM_TRIGGER_CHARS,
            "Alarm trigger too long: the stored trigger — the tool name plus its arguments as JSON — is limited to {MAX_ALARM_TRIGGER_CHARS} characters; shorten the arguments"
        );
    }

    // Normalize the one-shot fire time to RFC3339 UTC.
    let normalized_fire_at = fire_at
        .map(|f| {
            db::parse_utc_timestamp(f)
                .map(|dt| dt.to_rfc3339())
                .with_context(|| format!("Invalid RFC3339/ISO-8601 fire time: {f} — pass a UTC timestamp like 2026-08-28T10:30:00Z"))
        })
        .transpose()?;

    // Exactly one of fire_at / interval_seconds; a one-shot must be in the
    // future (a past timestamp is a caller error, never a fire-on-next-sweep).
    let (kind, interval_secs, next_fire_at) = match (&normalized_fire_at, interval_seconds) {
        (Some(fire), None) => {
            anyhow::ensure!(
                db::parse_utc_timestamp(fire)? > db::parse_utc_timestamp(&db::now())?,
                "Cannot set an alarm for a time in the past: pass a future RFC3339/ISO-8601 UTC time (convert the user's local time to UTC)"
            );
            ("one-shot", None, fire.clone())
        }
        (None, Some(interval)) => {
            anyhow::ensure!(
                interval >= 5,
                "Period must be at least 5 seconds: raise `interval_seconds`"
            );
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
        _ => anyhow::bail!(
            "Exactly one of fire_at or interval_seconds must be provided: pass `fire_at` for a one-shot reminder or `interval_seconds` for a periodic one, never both"
        ),
    };

    // Enforce the per-session active cap.
    let rows = store()
        .conn
        .query(
            "SELECT COUNT(*) FROM alarms WHERE session_id = ?1 AND status = 'active'",
            db::params![session_id.as_str()],
        )
        .await?;
    let count: i64 = rows.first().map(|r| r.get(0)).transpose()?.unwrap_or(0);
    anyhow::ensure!(
        count < MAX_ACTIVE_ALARMS,
        "Alarm limit reached (maximum {MAX_ACTIVE_ALARMS} active alarms): remove one with `remove_alarm` before adding another"
    );

    let id = crate::generate_id();
    let created_at = db::now();
    store()
        .conn
        .execute(
            "INSERT INTO alarms \
             (id, session_id, user_name, kind, text, fire_at, interval_seconds, next_fire_at, status, created_at, trigger, role, identifier, workspace) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'active', ?9, ?10, ?11, ?12, ?13)",
            db::params![
                id.clone(),
                session_id.as_str(),
                user_name,
                kind,
                text,
                normalized_fire_at.clone(),
                interval_secs,
                next_fire_at.clone(),
                created_at.clone(),
                encoded_trigger.clone(),
                role.clone(),
                identifier.clone(),
                workspace.clone(),
            ],
        )
        .await?;

    Ok(Alarm {
        id,
        session_id,
        user_name: user_name.to_string(),
        kind: kind.to_string(),
        text: text.to_string(),
        trigger: trigger.map(StoredTrigger::Tool),
        role,
        identifier,
        workspace,
        interval_seconds: interval_secs,
        next_fire_at,
        unfinished_attempts: 0,
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

/// Every active alarm of every owner, ordered by owner name, then next fire
/// time, then id — the read backing the alarms page. The id settles ties, so
/// rows sharing an owner and a fire time keep the same order from read to read.
/// The page groups by the name recorded on the alarm itself, so an alarm whose
/// account is gone is still listed under it.
pub(crate) async fn list_all_active_alarms() -> Result<Vec<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE status = 'active' \
         ORDER BY user_name ASC, next_fire_at ASC, id ASC"
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
    // The SELECT and the UPDATE are separate statements, so a concurrent second
    // removal of this alarm can pass the SELECT too and report it twice; the
    // UPDATE is idempotent (status stays 'removed'), so the duplicate report is
    // not a wrong outcome, and the alarm is removed exactly once.
    store()
        .conn
        .execute(
            "UPDATE alarms SET status = 'removed' WHERE id = ?1 AND session_id = ?2",
            db::params![id, session_id],
        )
        .await?;
    Ok(Some(alarm))
}

/// Mark every ACTIVE alarm addressed to `role_name` as `'removed'` (soft
/// delete, consistent with the rest of the module); returns the number changed.
#[expect(clippy::cast_possible_truncation)] // u64 row count fits usize on all supported targets
pub(crate) async fn delete_alarms_for_role(role_name: &str) -> Result<usize> {
    let changed = store()
        .conn
        .execute(
            "UPDATE alarms SET status = 'removed' WHERE role = ?1 AND status = 'active'",
            db::params![role_name],
        )
        .await?;
    Ok(changed as usize)
}

/// Soft-delete an alarm regardless of its current status (idempotent).
///
/// Unlike [`remove_alarm`] this has no `status = 'active'` gate, so it also
/// deletes an already-fired one-shot. Used whenever a check is broken — a run
/// that failed and one that could not run alike, never one that merely ran past
/// its time limit — so a broken alarm can never re-fire.
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

/// The due alarms at `now` (RFC3339 UTC), ordered by next fire time: every
/// occurrence that is due and whose retry cool-down, if it is waiting for one,
/// has passed. A waiting alarm is not due before its `retry_at` — and no other
/// alarm's wait holds it back.
///
/// The read here and each alarm's later state advance are separate statements,
/// so a second sweep overlapping this one can fire the same alarm twice; the
/// sweep runs in one background loop, and a triggered alarm's attempt slot
/// ([`claim_attempt_slot`]) makes a repeated firing of the same occurrence a
/// no-op.
async fn due_alarms(now: &str) -> Result<Vec<Alarm>> {
    let sql = format!(
        "SELECT {ALARM_COLUMNS} FROM alarms \
         WHERE status = 'active' AND next_fire_at <= ?1 \
         AND (retry_at IS NULL OR retry_at <= ?1) \
         ORDER BY next_fire_at ASC"
    );
    let rows = store().conn.query(&sql, db::params![now]).await?;
    Ok(rows.iter().map(alarm_from_row).collect::<Result<_, _>>()?)
}

// ── Fire / sweep ────────────────────────────────────────────────────────

/// Route a due alarm into the Assistant's own session and settle it.
///
/// A role-addressed alarm (`role` set) wakes that custom role instead; the
/// Assistant wakes otherwise — a triggered alarm (`trigger` set) takes its
/// attempt slot and runs its custom tool (see [`fire_trigger_alarm`]); a plain
/// alarm delivers the reminder text. The notification is delivered as a
/// user-role message into the calling assistant's own session
/// (`alarm.session_id`, resolved directly — never re-derived from the id, which
/// would double-escape colliding user names). A plain or role-addressed one-shot
/// is terminalized (`status='fired'`) and a periodic one advances `next_fire_at`
/// past every missed whole period; a triggered one settles the same way once its
/// check completes. A firing later than its scheduled time by more than
/// [`MISSED_FIRING_TOLERANCE`] says so in its notice (see [`reminder_line`]).
async fn fire_alarm(alarm: &Alarm) -> Result<()> {
    if alarm.role.is_some() {
        return fire_custom_role_alarm(alarm).await;
    }
    // Judged at the firing instant — the scheduled time as it stands before the
    // firing moves anything — never when the notice is composed: a slow check, or
    // a notice delivered late although the firing happened on time, is not a
    // missed firing.
    let missed = is_missed_firing(&db::now(), &alarm.next_fire_at);
    match &alarm.trigger {
        Some(trigger) => {
            fire_trigger_alarm(alarm, trigger, missed);
            Ok(())
        }
        None => fire_plain_alarm(alarm, missed).await,
    }
}

/// Wake the custom role an alarm is addressed to, with the alarm's text as the
/// launch message. A role that has since been deleted removes the alarm and
/// tells the owner through his Assistant: a broken alarm must never keep
/// firing. A store failure is not a verdict about the role — the alarm stays
/// armed and the next sweep retries.
///
/// Any other launch failure of a ONE-SHOT alarm is terminal: the alarm is
/// removed and the owner is told the reason. A PERIODIC alarm instead advances
/// its state like a completed fire and stays armed, so a transient failure (e.g.
/// a launch folder on a temporarily unavailable mount) loses no reminder and
/// retries at its next scheduled time rather than every sweep tick. The notice
/// names the reason and lets the owner re-arm once the cause is fixed.
async fn fire_custom_role_alarm(alarm: &Alarm) -> Result<()> {
    let role = alarm.role.as_deref().unwrap_or_default();
    // `removed` = this failure disarms the alarm; a periodic launch failure
    // keeps it armed (see below).
    let (reason, removed) = if crate::custom_role::get(role).await?.is_none() {
        ("the custom role no longer exists".to_string(), true)
    } else {
        // The launch runs in the folder the alarm was armed for; a folder that
        // no longer resolves is a launch failure like any other.
        let launched = async {
            let workspace = crate::launch::launch_workspace(alarm.workspace.as_deref()).await?;
            crate::launch::custom_role(
                role,
                &alarm.text,
                alarm.identifier.as_deref(),
                &workspace,
                crate::agent::message_router::AnsweredFor::Agent,
            )
            .await
        }
        .await;
        match launched {
            Ok(_) => {
                // The launch is already persisted durably before it is routed
                // (see `launch::custom_role`), so the state advance runs after,
                // exactly as on the plain path.
                advance_alarm_state(alarm, &db::now()).await?;
                return Ok(());
            }
            Err(e) if crate::db::is_store_error(&e) => {
                // A store failure says nothing about the role: leave the alarm
                // armed and let the next sweep retry, instead of disarming a
                // working reminder on a transient read/write error.
                tracing::warn!(
                    alarm = %alarm.id,
                    role = %role,
                    error = %e,
                    "Store failure launching a custom-role alarm — alarm left armed"
                );
                return Err(e);
            }
            Err(e) => {
                tracing::warn!(
                    alarm = %alarm.id,
                    role = %role,
                    error = %e,
                    "Failed to launch a custom role alarm"
                );
                if alarm.kind == "one-shot" {
                    (e.to_string(), true)
                } else {
                    // A periodic reminder must survive a transient launch
                    // failure: advance its state like a completed fire so it
                    // retries at its next scheduled time instead of every sweep
                    // tick (the sweep re-selects due alarms once a second).
                    advance_alarm_state(alarm, &db::now()).await?;
                    (e.to_string(), false)
                }
            }
        }
    };
    if removed {
        delete_alarm_after_run(alarm).await?;
    }
    // The notice goes to the owner (the assistant), never to the role.
    let workspace_line = match &alarm.workspace {
        Some(workspace) => format!(" in {workspace}"),
        None => String::new(),
    };
    let outcome = if removed {
        "The alarm was removed."
    } else {
        "The reminder stays armed and will fire again at its next scheduled time."
    };
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("alarm_role_notification.md"),
        &[
            ("{{text}}", &alarm.text),
            ("{{role}}", role),
            ("{{workspace_line}}", &workspace_line),
            ("{{reason}}", &reason),
            ("{{outcome}}", outcome),
        ],
    );
    deliver_alarm_notification(alarm, content).await;
    Ok(())
}

/// Deliver a plain (untriggered) alarm reminder and advance its state.
///
/// The durable envelope is persisted BEFORE advancing state (see
/// [`deliver_alarm_notification`]): a crash after persisting but before the
/// state advance replays the reminder at boot and leaves the alarm due, so the
/// reminder is never lost.
async fn fire_plain_alarm(alarm: &Alarm, missed: bool) -> Result<()> {
    let now = db::now();
    let content = crate::prompt::substitute(
        &crate::prompt::load_prompt("alarm_notification.md"),
        &[(
            "{{reminder_line}}",
            &reminder_line(alarm, &trigger_timing(alarm, missed, false)),
        )],
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
/// periodic → `next_fire_at` past every missed whole period, so a schedule that
/// waited out retries resumes on its own grid instead of replaying them. The
/// retry state goes with it: the attempt that got here completed. Both writes
/// take an active row only, so a reminder removed while its check ran stays
/// removed.
async fn advance_alarm_state(alarm: &Alarm, now: &str) -> Result<()> {
    if alarm.kind == "one-shot" {
        store()
            .conn
            .execute(
                "UPDATE alarms SET status = 'fired', retry_at = NULL, unfinished_attempts = 0 \
                 WHERE id = ?1 AND status = 'active'",
                db::params![alarm.id.as_str()],
            )
            .await?;
    } else {
        let next = next_periodic_fire(now, &alarm.next_fire_at, periodic_interval(alarm)?)?;
        store()
            .conn
            .execute(
                "UPDATE alarms SET next_fire_at = ?1, retry_at = NULL, unfinished_attempts = 0 \
                 WHERE id = ?2 AND status = 'active'",
                db::params![next, alarm.id.as_str()],
            )
            .await?;
    }
    Ok(())
}

/// The interval a periodic alarm advances by, refusing a stored row that has no
/// valid one instead of looping on it forever.
fn periodic_interval(alarm: &Alarm) -> Result<i64> {
    let interval = alarm.interval_seconds.unwrap_or(0);
    anyhow::ensure!(
        interval > 0,
        "Periodic alarm {} has no valid interval",
        alarm.id
    );
    Ok(interval)
}

/// Alarms whose check is running in this process, keyed by alarm id — the guard
/// that keeps an alarm to one attempt at a time.
static ATTEMPTS_IN_FLIGHT: LazyLock<std::sync::Mutex<HashSet<String>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

/// Take the alarm's one in-process attempt slot. `None` while a previous attempt
/// of the same alarm is still running; the returned slot frees it on drop —
/// including when a spawned task is dropped without ever having been polled,
/// because the slot is taken before the spawn. Nothing about it is durable: a
/// restarted process has no attempt in flight, so the alarm is simply due again.
fn claim_attempt_slot(alarm_id: &str) -> Option<AttemptSlot> {
    let inserted = ATTEMPTS_IN_FLIGHT
        .lock()
        .unwrap_poison()
        .insert(alarm_id.to_string());
    inserted.then(|| AttemptSlot(alarm_id.to_string()))
}

/// Whether the alarm's check is running in this process.
fn attempt_in_flight(alarm_id: &str) -> bool {
    ATTEMPTS_IN_FLIGHT.lock().unwrap_poison().contains(alarm_id)
}

/// RAII guard for an alarm's in-process attempt slot.
struct AttemptSlot(String);

impl Drop for AttemptSlot {
    fn drop(&mut self) {
        ATTEMPTS_IN_FLIGHT.lock().unwrap_poison().remove(&self.0);
    }
}

/// Fire a triggered alarm: take the alarm's attempt slot and spawn a detached
/// task that runs the stored check and settles the attempt.
///
/// The slot is taken before the spawn, so however slow the check is — and
/// whether or not the task ever gets polled — the sweep cannot start a second
/// attempt for the same alarm. The rest of the sweep is unaffected: a waiting
/// alarm holds back nothing but itself.
///
/// `missed` is the firing's lateness, decided before the check starts and
/// carried into the notice the finished run produces: how long the check itself
/// takes is not lateness.
fn fire_trigger_alarm(alarm: &Alarm, trigger: &StoredTrigger, missed: bool) {
    let Some(slot) = claim_attempt_slot(&alarm.id) else {
        return;
    };
    tokio::spawn(run_alarm_trigger_task(
        slot,
        alarm.clone(),
        trigger.clone(),
        missed,
    ));
}

/// What one trigger evaluation produced, as the notification and the alarm's
/// state see it.
enum TriggerOutcome {
    /// The check ran and reported nothing at all — the alarm stays armed and
    /// nothing is delivered.
    Silent,
    /// The check ran and reported something: the run's own outcome.
    Ran(ProgramOutcome),
    /// The check could not run at all: the reason the assistant is shown. The
    /// alarm goes either way.
    CouldNotRun(String),
    /// The check was still running when its own time limit stopped it. That is
    /// not a verdict about the check: nothing is delivered for the attempt, and
    /// the occurrence stays pending to be retried.
    Unfinished,
}

impl TriggerOutcome {
    /// The wake decision over one finished run: only a clean run that reported
    /// nothing at all stays silent. Everything else wakes the assistant — a run
    /// that failed, and output of any kind, whatever it looks like — except a run
    /// the ceiling stopped, which is settled by the retry rule instead.
    fn from_run(outcome: ProgramOutcome) -> Self {
        if outcome.timed_out {
            return Self::Unfinished;
        }
        if outcome.success && !outcome.has_output {
            Self::Silent
        } else {
            Self::Ran(outcome)
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
/// assistant is shown, and the run's own outcome — what it reported, and whether
/// the ceiling stopped it — alone drives the verdict
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
            ws.as_path(),
            &bun,
            &[call.path.display().to_string(), call.payload()],
        )
        .await,
    )
}

/// Run a triggered alarm's stored check in the owner's personal workspace and
/// settle the attempt (see [`settle_trigger_attempt`]). The attempt's slot is
/// held until this task finishes, however it finishes.
async fn run_alarm_trigger_task(
    _slot: AttemptSlot,
    alarm: Alarm,
    trigger: StoredTrigger,
    missed: bool,
) {
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

    if let Err(e) = settle_trigger_attempt(&alarm, &trigger, outcome, missed).await {
        // The occurrence is untouched: its due time still stands and no retry
        // cool-down was set, so the next sweep runs the check again.
        tracing::warn!(alarm = %alarm.id, error = %e, "Failed to settle an alarm check");
    }
}

/// Settle one finished attempt against its alarm.
///
/// A check that reported — or ran cleanly and said nothing — delivers what it
/// produced and lets the alarm resume its normal schedule. A broken check
/// soft-deletes the alarm and delivers why. An attempt that did not finish in
/// time delivers nothing and keeps the occurrence pending for a retry.
async fn settle_trigger_attempt(
    alarm: &Alarm,
    trigger: &StoredTrigger,
    outcome: TriggerOutcome,
    missed: bool,
) -> Result<()> {
    match &outcome {
        TriggerOutcome::Unfinished => return defer_trigger_attempt(alarm, trigger).await,
        TriggerOutcome::Silent | TriggerOutcome::Ran(ProgramOutcome { success: true, .. }) => {}
        TriggerOutcome::Ran(_) | TriggerOutcome::CouldNotRun(_) => {
            // Track whether the delete actually committed — the notification
            // must not claim a deletion that didn't happen.
            let deletion = delete_alarm_after_run(alarm).await;
            if let Some(content) =
                trigger_notification(alarm, trigger, &outcome, Some(&deletion), missed)
            {
                deliver_alarm_notification(alarm, content).await;
            }
            // A deletion that did not commit leaves the alarm due: the broken
            // check would run and notify again on every sweep. Advancing bounds
            // it to one more firing per scheduled occurrence, as every fire was
            // bounded before.
            if deletion.is_err() {
                advance_alarm_state(alarm, &db::now()).await?;
            }
            return Ok(());
        }
    }
    // Delivered before the state advance, so a crash in between replays the
    // report rather than losing it.
    if let Some(content) = trigger_notification(alarm, trigger, &outcome, None, missed) {
        deliver_alarm_notification(alarm, content).await;
    }
    advance_alarm_state(alarm, &db::now()).await
}

/// Keep the occurrence pending after an attempt that did not finish in time:
/// nothing is delivered for the attempt, the check is retried after the
/// cool-down, and the third attempt in a row says once that the check is not
/// producing a result.
///
/// `next_fire_at` is left alone — it is the occurrence's own due moment, which a
/// late report is announced with — so the wait lives in the row's `retry_at`. The
/// write is guarded on the occurrence the attempt ran for, so a concurrent
/// reschedule is never overwritten; the count is written as this attempt's own
/// value, which is the row's because an alarm has only one attempt in flight.
async fn defer_trigger_attempt(alarm: &Alarm, trigger: &StoredTrigger) -> Result<()> {
    let attempt = alarm.unfinished_attempts + 1;
    let retry_at = (db::parse_utc_timestamp(&db::now())?
        + ChronoDuration::seconds(RETRY_COOLDOWN_SECS))
    .to_rfc3339();
    let changed = store()
        .conn
        .execute(
            "UPDATE alarms SET retry_at = ?1, unfinished_attempts = ?2 \
             WHERE id = ?3 AND session_id = ?4 AND status = 'active' AND next_fire_at = ?5",
            db::params![
                retry_at,
                attempt,
                alarm.id.as_str(),
                alarm.session_id.as_str(),
                alarm.next_fire_at.as_str(),
            ],
        )
        .await?;
    if changed == 0 {
        return Ok(());
    }
    // The counter is what makes the notice one-time, so it is written before the
    // notice goes out: a crash in the gap loses one notice rather than repeating
    // it on every later unfinished attempt.
    if attempt == UNFINISHED_NOTICE_ATTEMPTS {
        let content = crate::prompt::substitute(
            &crate::prompt::load_prompt("alarm_trigger_unfinished.md"),
            &[
                ("{{text}}", &alarm.text),
                ("{{trigger}}", &trigger.render()),
            ],
        );
        deliver_alarm_notification(alarm, content).await;
    }
    Ok(())
}

/// Decide the delivery for a finished trigger evaluation: `None` stays silent
/// (the check reported nothing at all, or did not finish in time); `Some` is the
/// rendered `<alarm-notification>` with the reminder, the trigger, the status
/// line, the deletion note (on a broken check) and the check's output.
/// `deletion` carries the result of [`delete_alarm_after_run`], `None` when no
/// deletion was attempted; `missed` is the firing's lateness, decided when the
/// firing happened.
fn trigger_notification(
    alarm: &Alarm,
    trigger: &StoredTrigger,
    outcome: &TriggerOutcome,
    deletion: Option<&Result<()>>,
    missed: bool,
) -> Option<String> {
    let (status, output) = match outcome {
        TriggerOutcome::Silent | TriggerOutcome::Unfinished => return None,
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
    // Only a delivered result can be late for its occurrence — a broken check's
    // notice reads as it always has.
    let reported = matches!(outcome, TriggerOutcome::Ran(run) if run.success);
    let timing = trigger_timing(alarm, missed, reported);
    Some(crate::prompt::substitute(
        &crate::prompt::load_prompt("alarm_trigger_notification.md"),
        &[
            ("{{reminder_line}}", &reminder_line(alarm, &timing)),
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

/// The alarms a sweep pass may start, up to `batch_limit`: those due and not
/// already running here. A running check is skipped before the batch is cut, so
/// a long check never takes a slot from a waiting alarm; the alarm's own slot
/// ([`claim_attempt_slot`]) stays the authority on never starting twice.
fn startable(due: Vec<Alarm>, batch_limit: usize) -> Vec<Alarm> {
    due.into_iter()
        .filter(|alarm| !attempt_in_flight(&alarm.id))
        .take(batch_limit)
        .collect()
}

/// Fire up to `batch_limit` due alarms, continuing past individual failures so
/// one bad alarm cannot block the rest of the sweep.
async fn run_alarm_sweep(batch_limit: usize) -> Result<()> {
    let due = startable(due_alarms(&db::now()).await?, batch_limit);
    let mut fired = 0usize;
    let mut failed = 0usize;
    for alarm in due {
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
        if !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_TICK).await {
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

    /// A firing is MISSED only when it lands well past its scheduled time: a
    /// routine tick's own delay never counts, and a scheduled time that cannot
    /// be read is not a miss either — that would fail the firing of an alarm
    /// that stays due, re-firing it every tick.
    #[test]
    fn missed_firing_needs_a_firing_far_past_its_scheduled_time() {
        let scheduled = "2026-09-04T12:00:00+00:00";
        assert!(!is_missed_firing("2026-09-04T12:00:00+00:00", scheduled));
        assert!(!is_missed_firing("2026-09-04T12:00:05+00:00", scheduled));
        assert!(!is_missed_firing("2026-09-04T12:00:30+00:00", scheduled));
        assert!(
            is_missed_firing("2026-09-04T12:01:00+00:00", scheduled),
            "a firing a minute late is past the tolerance"
        );
        assert!(
            is_missed_firing("2026-09-04T23:00:00+00:00", scheduled),
            "an overnight gap must be reported as missed"
        );
        assert!(!is_missed_firing("2026-09-04T12:00:00+00:00", "not-a-time"));
        assert!(!is_missed_firing("2026-09-04T11:00:00+00:00", scheduled));
    }

    /// The opening line is what the notice says about its own timing: today's
    /// wording when the firing happened on time, the missed statement with the
    /// scheduled time when it fired late, and the late statement — one of the
    /// two, never both — naming the occurrence when a retried check finally
    /// reported.
    #[test]
    fn reminder_line_states_exactly_one_timing() {
        let alarm = trigger_alarm();
        assert_eq!(
            reminder_line(&alarm, &Timing::Due),
            "Your reminder \"check the thing\" is due now \
             (scheduled for 2026-09-04T12:00:00+00:00)."
        );
        let missed = reminder_line(&alarm, &Timing::Missed);
        assert!(missed.contains("was MISSED"), "got: {missed}");
        assert!(
            missed.contains("2026-09-04T12:00:00+00:00"),
            "the scheduled time must be named: {missed}"
        );
        assert!(
            missed.contains("Tell the user"),
            "the assistant must be told to pass it on: {missed}"
        );
        assert!(
            !missed.contains("due now"),
            "a missed firing must not read as current: {missed}"
        );

        let due_at = "2026-09-04T12:00:00+00:00";
        let late = reminder_line(&alarm, &Timing::Late);
        assert!(late.contains("LATE"), "got: {late}");
        assert!(
            late.contains(due_at),
            "the moment it was due for must be named: {late}"
        );
        assert!(
            !late.contains("MISSED") && !late.contains("due now"),
            "a late report is one statement, not a stack of lateness: {late}"
        );
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
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("past"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_rejects_short_periodic_interval() {
        let err = add_alarm("session-a", "alice", "remind me", None, Some(4), None, None)
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
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("at most 292 years"), "got: {err}");
    }

    #[tokio::test]
    async fn add_alarm_requires_exactly_one_of_fire_at_or_interval() {
        // Neither provided.
        let err = add_alarm("session-a", "alice", "remind me", None, None, None, None)
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
        for i in 0..MAX_ACTIVE_ALARMS {
            let fire = format!("2099-01-01T00:00:{i:02}Z");
            add_alarm(
                session,
                "alice",
                &format!("reminder {i}"),
                Some(&fire),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        }
        let err = add_alarm(
            session,
            "alice",
            "one too many",
            Some("2099-01-01T00:01:00Z"),
            None,
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains(&format!("maximum {MAX_ACTIVE_ALARMS}")),
            "got: {err}"
        );
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
                None,
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
            None,
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
            trigger: Some(StoredTrigger::Tool(weather_trigger())),
            role: None,
            identifier: None,
            workspace: None,
            interval_seconds: Some(60),
            next_fire_at: "2026-09-04T12:00:00+00:00".to_string(),
            unfinished_attempts: 0,
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
            timed_out: false,
        }
    }

    #[test]
    fn trigger_notification_stays_silent_when_check_reports_nothing() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        for missed in [false, true] {
            assert!(
                trigger_notification(&alarm, &trigger, &TriggerOutcome::Silent, None, missed)
                    .is_none(),
                "a silent check must not wake the assistant, late or not"
            );
        }
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
        let content = trigger_notification(&alarm, &trigger, &outcome, None, false).unwrap();
        assert!(content.contains("<alarm-notification>"));
        assert!(content.contains("check the thing"));
        assert!(content.contains("weather {\"city\":\"Minsk\"}"));
        assert!(content.contains("exited successfully (exit status 0)"));
        assert!(content.contains("all good"));
        assert!(
            !content.contains("DELETED"),
            "clean path must not report a deletion"
        );
        assert!(
            content.contains("is due now (scheduled for 2026-09-04T12:00:00+00:00)."),
            "a firing on time keeps today's wording: {content}"
        );
    }

    /// A late firing is reported as missed, naming the time it was scheduled
    /// for, on every notice a firing produces.
    #[test]
    fn trigger_notification_says_a_late_firing_was_missed() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        for outcome in [
            TriggerOutcome::Ran(ran(true, "exit status 0", "all good", true)),
            TriggerOutcome::Ran(ran(false, "exit status 2", "boom", true)),
            TriggerOutcome::CouldNotRun("the managed bun runtime is unavailable".to_string()),
        ] {
            let content = trigger_notification(&alarm, &trigger, &outcome, None, true).unwrap();
            assert!(content.contains("was MISSED"), "got: {content}");
            assert!(
                content.contains("2026-09-04T12:00:00+00:00"),
                "got: {content}"
            );
            assert!(!content.contains("due now"), "got: {content}");
        }
    }

    /// Output that strips away to nothing is still reported, so the notice must
    /// not label it "no output".
    #[test]
    fn trigger_notification_does_not_call_escape_only_output_empty() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let outcome = TriggerOutcome::Ran(ran(true, "exit status 0", "", true));
        let content = trigger_notification(&alarm, &trigger, &outcome, None, false).unwrap();
        assert!(content.contains("not printable"), "got: {content}");
        assert!(!content.contains("(no output)"), "got: {content}");
    }

    /// A broken check's notice reads as it always has, even when it settles an
    /// occurrence that waited through retries: no lateness note — only a report
    /// can be late — and the moment it names is the occurrence's own scheduled
    /// time, never the retry instant the notice is composed at.
    #[test]
    fn a_broken_check_after_retries_is_not_late_and_names_its_scheduled_time() {
        let alarm = Alarm {
            unfinished_attempts: 3,
            ..trigger_alarm()
        };
        let trigger = StoredTrigger::Tool(weather_trigger());
        for outcome in [
            TriggerOutcome::Ran(ran(false, "exit status 2", "boom", true)),
            TriggerOutcome::CouldNotRun("the managed bun runtime is unavailable".to_string()),
        ] {
            let content = trigger_notification(&alarm, &trigger, &outcome, Some(&Ok(())), true)
                .expect("a broken check always notifies");
            assert!(!content.contains("LATE"), "got: {content}");
            assert!(
                content.contains("was MISSED — it was scheduled for 2026-09-04T12:00:00+00:00"),
                "the occurrence's own scheduled time must be named: {content}"
            );
        }
    }

    /// A failed check always wakes, empty output included, and the notice
    /// reports whether the alarm was actually removed.
    #[test]
    fn trigger_notification_wakes_on_failure_and_reports_the_deletion_outcome() {
        let alarm = trigger_alarm();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let outcome = TriggerOutcome::Ran(ran(false, "exit status 2", "", false));

        let deleted =
            trigger_notification(&alarm, &trigger, &outcome, Some(&Ok(())), false).unwrap();
        assert!(deleted.contains("FAILED (exit status 2)"), "got: {deleted}");
        assert!(deleted.contains("(no output)"), "got: {deleted}");
        assert!(deleted.contains("DELETED"), "got: {deleted}");

        let failed = trigger_notification(
            &alarm,
            &trigger,
            &outcome,
            Some(&Err(anyhow::anyhow!("db down"))),
            false,
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
        let content =
            trigger_notification(&alarm, &trigger, &outcome, Some(&Ok(())), false).unwrap();
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

    /// The moment the alarm's retried occurrence may run, `None` when nothing
    /// is waiting — read straight from the row, as the sweep's own filter is.
    async fn alarm_retry_at(id: &str) -> Option<String> {
        store()
            .conn
            .query("SELECT retry_at FROM alarms WHERE id = ?1", db::params![id])
            .await
            .unwrap()
            .first()
            .map(|row| row.get::<Option<String>>(0))
            .transpose()
            .unwrap()
            .flatten()
    }

    /// The row as the next sweep would read it.
    async fn active_alarm(session_id: &str, id: &str) -> Alarm {
        list_alarms(session_id)
            .await
            .unwrap()
            .into_iter()
            .find(|alarm| alarm.id == id)
            .expect("an active alarm row")
    }

    /// Whether the sweep would pick the alarm up right now.
    async fn is_due(id: &str) -> bool {
        due_alarms(&db::now())
            .await
            .unwrap()
            .iter()
            .any(|alarm| alarm.id == id)
    }

    /// How far `timestamp` is from now, in seconds.
    fn seconds_from_now(timestamp: &str) -> i64 {
        db::parse_utc_timestamp(timestamp)
            .unwrap()
            .signed_duration_since(db::parse_utc_timestamp(&db::now()).unwrap())
            .num_seconds()
    }

    /// Make the alarm's occurrence overdue, as a sweep's read would find it.
    async fn make_overdue(id: &str) {
        store()
            .conn
            .execute(
                "UPDATE alarms SET next_fire_at = ?1 WHERE id = ?2",
                db::params!["2020-01-01T00:00:00+00:00", id],
            )
            .await
            .unwrap();
    }

    /// One attempt at a time, and a waiting alarm holds back nothing but itself:
    /// an alarm whose check is running has no second attempt, a deferred
    /// occurrence is not due before its cool-down, and every other alarm keeps
    /// firing at its own due time meanwhile.
    #[tokio::test]
    async fn one_attempt_at_a_time_and_a_waiting_alarm_waits_only_for_itself() {
        let session = "assistant:alarm-one-attempt-it";
        let (waiting, _rx) = arm_trigger_alarm(
            session,
            "alarm-waiting",
            "no_such_alarm_tool",
            serde_json::Map::new(),
        )
        .await;
        let neighbour = add_alarm(
            session,
            "alarm-neighbour",
            "the other check",
            None,
            Some(3600),
            Some(weather_trigger()),
            None,
        )
        .await
        .expect("arm a second triggered alarm");
        make_overdue(&waiting.id).await;
        make_overdue(&neighbour.id).await;
        assert!(
            is_due(&waiting.id).await && is_due(&neighbour.id).await,
            "both occurrences are due"
        );

        // The attempt in flight holds the alarm's only slot: a second one never
        // starts, however long the check runs.
        let slot = claim_attempt_slot(&waiting.id).expect("the attempt takes the alarm's slot");
        assert!(
            claim_attempt_slot(&waiting.id).is_none(),
            "a second attempt of the same alarm must not start"
        );
        drop(slot);

        let due = active_alarm(session, &waiting.id).await;
        settle_trigger_attempt(
            &due,
            &StoredTrigger::Tool(weather_trigger()),
            TriggerOutcome::Unfinished,
            false,
        )
        .await
        .unwrap();
        let deferred = active_alarm(session, &waiting.id).await;
        let retry_at = alarm_retry_at(&waiting.id)
            .await
            .expect("a wait is recorded");
        let retry_in = seconds_from_now(&retry_at);
        assert!(
            (50..=70).contains(&retry_in),
            "the retry must be about a minute out, got {retry_in}s"
        );
        assert_eq!(
            deferred.next_fire_at, due.next_fire_at,
            "the wait never moves the occurrence's own due moment"
        );
        assert!(
            !is_due(&waiting.id).await,
            "a waiting alarm is not due before its retry"
        );
        assert!(
            is_due(&neighbour.id).await,
            "a waiting alarm must not hold back the others"
        );

        // Once the cool-down has passed the occurrence is due again — with no
        // state but the row's, so a restart in between changes nothing.
        store()
            .conn
            .execute(
                "UPDATE alarms SET retry_at = ?1 WHERE id = ?2",
                db::params!["2020-01-01T00:00:00+00:00", waiting.id.as_str()],
            )
            .await
            .unwrap();
        assert!(
            is_due(&waiting.id).await,
            "the retry becomes due when its cool-down passes"
        );
        crate::agent::message_router::unregister_agent(session);
    }

    /// The whole chain, from a program the ceiling really stops to the notice
    /// the retry finally produces: the stop keeps the alarm (nothing delivered,
    /// a retry about a minute out, the occurrence's own due moment intact) and
    /// the report of the later attempt arrives late, naming that moment.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_check_stopped_by_its_ceiling_keeps_the_alarm_and_its_report_is_late() {
        let session = "assistant:alarm-ceiling-it";
        let (alarm, mut rx) = arm_trigger_alarm(
            session,
            "alarm-ceiling",
            "no_such_alarm_tool",
            serde_json::Map::new(),
        )
        .await;
        let due_at = alarm.next_fire_at.clone();
        let trigger = StoredTrigger::Tool(weather_trigger());
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let ws = crate::workspace::test_ws(tmp.path());

        let stopped = crate::tools::shell::run_program_outcome_within(
            ws.as_path(),
            std::path::Path::new("/bin/sh"),
            &["-c".to_string(), "echo started; sleep 30".to_string()],
            Duration::from_secs(1),
        )
        .await;
        assert!(
            stopped.timed_out,
            "the one-second ceiling must stop the run"
        );
        settle_trigger_attempt(&alarm, &trigger, TriggerOutcome::from_run(stopped), false)
            .await
            .unwrap();

        assert!(
            rx.try_recv().is_err(),
            "nothing is delivered for an attempt that did not finish"
        );
        // A restart between the two attempts keeps everything the report needs:
        // the wait lives in the row read back here, and a fresh process has no
        // attempt in flight.
        assert!(
            !attempt_in_flight(&alarm.id),
            "a restarted process has no attempt in flight"
        );
        let waiting = active_alarm(session, &alarm.id).await;
        assert_eq!(
            alarm_status(&alarm.id).await,
            "active",
            "a one-shot whose check did not finish is not consumed"
        );
        assert_eq!(waiting.unfinished_attempts, 1);
        assert_eq!(waiting.next_fire_at, due_at);
        assert!(
            seconds_from_now(&alarm_retry_at(&alarm.id).await.expect("a wait is recorded")) > 0,
            "the occurrence is kept and its retry is scheduled"
        );

        // The retry reports: one notification, late, for the occurrence's own
        // due moment. A retried firing is past the missed-firing tolerance too,
        // so the two lateness statements must not stack.
        settle_trigger_attempt(
            &waiting,
            &trigger,
            TriggerOutcome::Ran(ran(true, "exit status 0", "found it", true)),
            true,
        )
        .await
        .unwrap();
        let job = rx.try_recv().expect("the late report must be delivered");
        assert!(job.content.contains("LATE"), "got: {}", job.content);
        assert!(
            !job.content.contains("MISSED"),
            "one lateness statement, not two: {}",
            job.content
        );
        assert!(
            job.content.contains(&due_at),
            "the moment it was due for must be named: {}",
            job.content
        );
        assert!(job.content.contains("found it"));
        crate::agent::message_router::unregister_agent(session);
    }

    /// A failed run is a broken check, not a check that did not finish: it
    /// removes the alarm, says FAILED, and leaves nothing due to re-fire.
    #[tokio::test]
    async fn a_failed_check_removes_the_alarm_and_never_refires() {
        let session = "assistant:alarm-failed-it";
        let (alarm, mut rx) = arm_trigger_alarm(
            session,
            "alarm-failed",
            "no_such_alarm_tool",
            serde_json::Map::new(),
        )
        .await;
        make_overdue(&alarm.id).await;
        let due = active_alarm(session, &alarm.id).await;

        settle_trigger_attempt(
            &due,
            &StoredTrigger::Tool(weather_trigger()),
            TriggerOutcome::Ran(ran(false, "exit status 2", "boom", true)),
            false,
        )
        .await
        .unwrap();

        let job = rx.try_recv().expect("a broken check must notify");
        assert!(
            job.content.contains("FAILED (exit status 2)"),
            "got: {}",
            job.content
        );
        assert!(job.content.contains("DELETED"), "got: {}", job.content);
        assert_eq!(alarm_status(&alarm.id).await, "removed");
        assert!(
            !is_due(&alarm.id).await,
            "a broken check must never re-fire"
        );
        crate::agent::message_router::unregister_agent(session);
    }

    /// A sweep pass skips an alarm whose check is already running before it cuts
    /// its batch, so a long check never takes a slot from a waiting alarm; the
    /// alarm's own slot stays the authority on not starting twice.
    #[test]
    fn a_running_check_does_not_take_a_sweep_slot() {
        let running = Alarm {
            id: "alarm-slot-running".to_string(),
            ..trigger_alarm()
        };
        let waiting = Alarm {
            id: "alarm-slot-waiting".to_string(),
            ..trigger_alarm()
        };
        let slot = claim_attempt_slot(&running.id).expect("the running check holds its slot");

        let picked = startable(vec![running.clone(), waiting.clone()], 1);
        assert_eq!(picked.len(), 1, "the batch keeps its full size");
        assert_eq!(
            picked[0].id, waiting.id,
            "the running check is skipped, not counted"
        );
        assert!(
            !attempt_in_flight(&waiting.id),
            "picking an alarm takes no slot; its firing does"
        );
        drop(slot);
    }

    /// Three attempts in a row that did not finish wake the assistant once, with
    /// a notice that says the check is not producing a result — and the alarm
    /// keeps being retried without repeating the notice until an attempt
    /// completes, which starts the count over.
    #[tokio::test]
    async fn three_unfinished_attempts_notify_once_and_the_alarm_keeps_retrying() {
        let session = "assistant:alarm-stuck-it";
        let trigger = StoredTrigger::Tool(weather_trigger());
        let mut alarm = arm_periodic_trigger_alarm(session, "alarm-stuck", 3600).await;
        let mut rx = crate::agent::message_router::register_agent(session);

        for attempt in 1..=4 {
            settle_trigger_attempt(&alarm, &trigger, TriggerOutcome::Unfinished, false)
                .await
                .unwrap();
            alarm = active_alarm(session, &alarm.id).await;
            assert_eq!(alarm.unfinished_attempts, attempt);
            if attempt == UNFINISHED_NOTICE_ATTEMPTS {
                let job = rx
                    .try_recv()
                    .expect("the third unfinished attempt must notify");
                assert!(
                    job.content.contains("not producing a result"),
                    "got: {}",
                    job.content
                );
                assert!(
                    !job.content.contains("DELETED"),
                    "this is not the deletion notice: {}",
                    job.content
                );
            } else {
                assert!(rx.try_recv().is_err(), "attempt {attempt} must stay silent");
            }
        }
        assert_eq!(
            alarm_status(&alarm.id).await,
            "active",
            "a check that never finishes never removes its alarm"
        );

        // A completing attempt clears the count, so the notice needs three more
        // unfinished attempts in a row and does not fire again before that.
        settle_trigger_attempt(&alarm, &trigger, TriggerOutcome::Silent, false)
            .await
            .unwrap();
        alarm = active_alarm(session, &alarm.id).await;
        assert_eq!(alarm.unfinished_attempts, 0);
        assert!(
            alarm_retry_at(&alarm.id).await.is_none(),
            "a completed check leaves no wait"
        );
        for attempt in 1..UNFINISHED_NOTICE_ATTEMPTS {
            settle_trigger_attempt(&alarm, &trigger, TriggerOutcome::Unfinished, false)
                .await
                .unwrap();
            alarm = active_alarm(session, &alarm.id).await;
            assert_eq!(alarm.unfinished_attempts, attempt, "the count restarted");
            assert!(
                rx.try_recv().is_err(),
                "the count restarted, so no notice arrives yet"
            );
        }
        crate::agent::message_router::unregister_agent(session);
    }

    /// A periodic alarm whose check completes resumes its normal schedule from
    /// the occurrence the retries settled: no burst of the occurrences missed
    /// while it waited, and no stale retry state left behind.
    #[tokio::test]
    async fn a_completed_check_resumes_the_periodic_schedule() {
        let session = "assistant:alarm-resume-it";
        let alarm = arm_periodic_trigger_alarm(session, "alarm-resume", 60).await;
        let mut rx = crate::agent::message_router::register_agent(session);
        let trigger = StoredTrigger::Tool(weather_trigger());
        // Long overdue, as an occurrence that waited out retries is: the resume
        // has to advance past every period missed meanwhile in one step.
        make_overdue(&alarm.id).await;
        let due = active_alarm(session, &alarm.id).await;
        let due_at = due.next_fire_at.clone();

        settle_trigger_attempt(&due, &trigger, TriggerOutcome::Unfinished, false)
            .await
            .unwrap();
        let retry = active_alarm(session, &alarm.id).await;
        assert!(
            alarm_retry_at(&alarm.id).await.is_some(),
            "a waiting alarm records the retry it owes"
        );
        settle_trigger_attempt(&retry, &trigger, TriggerOutcome::Silent, false)
            .await
            .unwrap();

        let resumed = active_alarm(session, &alarm.id).await;
        assert!(alarm_retry_at(&alarm.id).await.is_none());
        assert_eq!(resumed.unfinished_attempts, 0);
        let next_in = seconds_from_now(&resumed.next_fire_at);
        assert!(
            (0..=60).contains(&next_in),
            "the next occurrence is one period out at most, got {next_in}s"
        );
        let due = db::parse_utc_timestamp(&due_at).unwrap();
        let next = db::parse_utc_timestamp(&resumed.next_fire_at).unwrap();
        assert_eq!(
            next.signed_duration_since(due).num_seconds() % 60,
            0,
            "the schedule resumes on its own grid: {} from {due_at}",
            resumed.next_fire_at
        );
        assert!(
            rx.try_recv().is_err(),
            "a silent check must not wake the assistant"
        );
        crate::agent::message_router::unregister_agent(session);
    }

    /// Arm a periodic trigger alarm for `owner`, firing every `period_secs`,
    /// under the agent `session_id`.
    async fn arm_periodic_trigger_alarm(session_id: &str, owner: &str, period_secs: u64) -> Alarm {
        seed_owner(owner).await;
        add_alarm(
            session_id,
            owner,
            "poll the thing",
            None,
            Some(period_secs),
            Some(weather_trigger()),
            None,
        )
        .await
        .expect("arm a periodic trigger alarm")
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
            None,
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
            None,
        )
        .await
        .unwrap();

        fire_alarm(&alarm).await.unwrap();

        let job = rx.recv().await.expect("plain reminder must route");
        assert!(job.content.contains("<alarm-notification>"));
        assert!(job.content.contains("check the deploy"));
        assert!(
            !job.content.contains("MISSED"),
            "a firing on time must not claim it was missed: {}",
            job.content
        );
        crate::agent::message_router::unregister_agent("assistant:alarm-plain-it");

        assert_eq!(
            alarm_status(&alarm.id).await,
            "fired",
            "one-shot must be terminalized"
        );
    }

    /// Fire-time integration: an alarm the sweep reaches long after its
    /// scheduled time — a start after the product was down, a machine that was
    /// suspended — reaches the user as missed, naming the occurrence that was
    /// due rather than the next one the state advance has already written.
    #[tokio::test]
    async fn fire_late_alarm_says_the_firing_was_missed() {
        crate::util::test::init_management_test_stores().await;
        let session = "assistant:alarm-late-it";
        let mut rx = crate::agent::message_router::register_agent(session);
        let alarm = add_alarm(
            session,
            "alarm-late",
            "check the deploy",
            None,
            Some(3600),
            None,
            None,
        )
        .await
        .unwrap();
        // Make it due with its scheduled time long past, and fire the row the
        // sweep's own read selects.
        store()
            .conn
            .execute(
                "UPDATE alarms SET next_fire_at = ?1 WHERE id = ?2",
                db::params!["2020-01-01T00:00:00Z", alarm.id.as_str()],
            )
            .await
            .unwrap();
        let due = due_alarms(&db::now()).await.unwrap();
        let stored = due
            .iter()
            .find(|candidate| candidate.id == alarm.id)
            .expect("the alarm must be due");

        fire_alarm(stored).await.unwrap();

        let job = rx.recv().await.expect("late reminder must route");
        crate::agent::message_router::unregister_agent(session);
        assert!(
            job.content
                .contains("scheduled for 2020-01-01T00:00:00Z, which had already passed"),
            "the occurrence that was missed must be the one named: {}",
            job.content
        );
        assert_eq!(
            alarm_status(&alarm.id).await,
            "active",
            "a periodic alarm survives its firing"
        );
    }

    /// A role target with no branch and no workspace.
    fn role_target(role: &str) -> AlarmTarget {
        AlarmTarget {
            role: role.to_string(),
            identifier: None,
            workspace: None,
        }
    }

    /// The refusal from arming a role-addressed alarm under `user`.
    async fn try_arm_role(user: &str, role: &str, trigger: Option<Trigger>) -> String {
        add_alarm(
            "assistant:role-target-it",
            user,
            "run it",
            Some("2099-01-01T00:00:00Z"),
            None,
            trigger,
            Some(role_target(role)),
        )
        .await
        .unwrap_err()
        .to_string()
    }

    /// Arming a role-addressed alarm fails closed: only the admin may address a
    /// role, the role must exist, and the alarm's text (the role's launch
    /// message) cannot also carry a trigger.
    #[tokio::test]
    async fn add_alarm_validates_a_role_target_before_storing_it() {
        crate::util::test::init_test_stores().await;

        let err = try_arm_role("alice", "analyst", None).await;
        assert!(err.contains("only the administrator"), "got: {err}");

        let err = try_arm_role(crate::users::ADMIN_USER_NAME, "no_such_role", None).await;
        assert!(
            err.contains("no custom role named 'no_such_role'"),
            "got: {err}"
        );

        crate::custom_role::create(
            "alarm_role_probe",
            "do it",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let err = try_arm_role(
            crate::users::ADMIN_USER_NAME,
            "alarm_role_probe",
            Some(weather_trigger()),
        )
        .await;
        assert!(
            err.contains("cannot be combined with a role target"),
            "got: {err}"
        );
    }

    /// Fire-time integration: an alarm addressed to a custom role launches that
    /// role's branch with the alarm's text as its launch message, then
    /// terminalizes the one-shot.
    #[tokio::test]
    async fn fire_role_alarm_launches_the_role_and_terminalizes_the_alarm() {
        crate::util::test::init_management_test_stores().await;
        crate::custom_role::create(
            "alarm_runner",
            "do the thing",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let alarm = add_alarm(
            &crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME),
            crate::users::ADMIN_USER_NAME,
            "run the runner",
            Some("2099-01-01T00:00:00Z"),
            None,
            None,
            Some(role_target("alarm_runner")),
        )
        .await
        .unwrap();

        let branch = crate::session::custom_role_agent_id("alarm_runner", None);
        let mut rx = crate::agent::message_router::register_agent(&branch);
        fire_alarm(&alarm).await.unwrap();
        let job = rx
            .recv()
            .await
            .expect("the role's branch must receive the launch message");
        crate::agent::message_router::unregister_agent(&branch);
        assert_eq!(job.content, "run the runner");
        assert_eq!(
            alarm_status(&alarm.id).await,
            "fired",
            "one-shot must be terminalized"
        );
    }

    /// Fire-time integration: a role deleted after arming takes its alarm with
    /// it — the owner is told through his Assistant and the alarm is removed.
    #[tokio::test]
    async fn fire_role_alarm_reports_a_deleted_role_and_removes_the_alarm() {
        crate::util::test::init_management_test_stores().await;
        crate::custom_role::create(
            "alarm_vanishing",
            "do the thing",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let alarm = add_alarm(
            &crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME),
            crate::users::ADMIN_USER_NAME,
            "wake the role",
            Some("2099-01-01T00:00:00Z"),
            None,
            None,
            Some(role_target("alarm_vanishing")),
        )
        .await
        .unwrap();
        // Remove the record out of band, leaving the armed alarm behind.
        crate::custom_role::CUSTOM_ROLES
            .get()
            .expect("custom role store")
            .conn
            .execute(
                "DELETE FROM custom_roles WHERE name = ?1",
                db::params!["alarm_vanishing"],
            )
            .await
            .unwrap();

        let mut rx = crate::agent::message_router::register_agent(&alarm.session_id);
        fire_alarm(&alarm).await.unwrap();
        let job = rx.recv().await.expect("the owner must be told");
        crate::agent::message_router::unregister_agent(&alarm.session_id);
        assert!(job.content.contains("<alarm-notification>"));
        assert!(
            job.content.contains("alarm_vanishing"),
            "got: {}",
            job.content
        );
        assert!(
            job.content.contains("no longer exists"),
            "got: {}",
            job.content
        );
        assert_eq!(alarm_status(&alarm.id).await, "removed");
    }

    /// `delete_alarms_for_role` soft-deletes exactly the role's ACTIVE alarms.
    #[tokio::test]
    async fn delete_alarms_for_role_soft_deletes_only_its_active_alarms() {
        crate::util::test::init_management_test_stores().await;
        crate::custom_role::create(
            "alarm_cleanup",
            "done",
            &[],
            crate::custom_role::CustomRoleBehavior::Continuous,
        )
        .await
        .unwrap();
        let alarm = add_alarm(
            &crate::session::assistant_agent_id(crate::users::ADMIN_USER_NAME),
            crate::users::ADMIN_USER_NAME,
            "one",
            Some("2099-01-01T00:00:00Z"),
            None,
            None,
            Some(role_target("alarm_cleanup")),
        )
        .await
        .unwrap();
        // A plain alarm of another owner must survive.
        let plain = add_alarm(
            "assistant:alarm-cleanup-it",
            "alarm-cleanup-user",
            "two",
            Some("2099-01-01T00:00:00Z"),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(delete_alarms_for_role("alarm_cleanup").await.unwrap(), 1);
        assert_eq!(alarm_status(&alarm.id).await, "removed");
        assert_eq!(alarm_status(&plain.id).await, "active");
        // Idempotent: nothing left to remove.
        assert_eq!(delete_alarms_for_role("alarm_cleanup").await.unwrap(), 0);
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
