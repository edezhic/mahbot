//! The service's ONE durable failure record: `<root>/error.log`.
//!
//! Every severe failure the service survives — or refuses to survive — is appended here as a
//! block: a start-up refusal, any other store bring-up failure, a launch failure, a persistent
//! periodic or exit-time checkpoint failure, a runtime integrity failure, and a refused
//! reclaiming-checkpoint shrink. [`record`] is the single write entry point: with a resolvable
//! storage root the block lands in the file, and without one (or when the write fails) the full
//! block goes to stderr — a fallback a launch given no standard streams cannot observe.
//!
//! Callers render a [`FailureReport`] and hand it to [`record`]; the raw append lives in
//! [`append_failure_record`]. One file, one format, one place that knows how to append a block
//! without tearing.
//!
//! That entry point is bounded: [`record`] appends only when the record does not already hold a
//! block for the same CONDITION — the identity lines ([`condition_lines`]), what it is about
//! rather than what it says. Two occurrences worded differently (a size in the reason, a localised
//! engine message) are one block, and so is a second, genuinely different failure of the same kind
//! on the same artifact — the price of a bound that does not hinge on wording — while a different
//! store, step or cause class still files its own. Because the bound lives in the writer, no
//! caller can bypass it.
//!
//! This is the durable half of the rule the repeating in-process kinds follow ([`RoundCounter`]:
//! file the first round, then only count). A launch that exits has no process to count in, so the
//! record itself is the count, keyed on the kind and the artifact or step rather than the wording.
//!
//! A checkpoint failure's block carries the engine's own reason on its `reason:` line
//! ([`crate::db::checkpoint_cause`]): only the engine can fail the checkpoint, since a store
//! carries one serialised writing connection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use crate::util::UnwrapPoison;

const ERROR_LOG_NAME: &str = "error.log";

/// The two artifact lines a block that names no store renders: a config load, a provider or another
/// global init has no store and no db path to name, and the block says so rather than omitting
/// either line — a claim about what this block carries, never about the cause, since a panic inside
/// a store bring-up reaches this shape too (the binary's `bootstrap` step). Every kind whose failure
/// IS about a store fills both fields ([`FailureReport::store`] / [`FailureReport::db_path`]).
pub(crate) const NO_STORE: &str = "store: none (this block names no store)";
pub(crate) const NO_DB_PATH: &str = "db path: none (this block names neither)";

/// The artifact-state line's not-obtainable wording (see [`artifact_state_line`]). Kept neutral
/// about who failed to measure: the callers that reach it are a failed stat and a probe that
/// answered nothing at all.
const ARTIFACT_STATE_UNAVAILABLE: &str = "artifact state: not obtainable (no -wal size measured)";

/// The artifact state every per-store block reports: the size of the store's `-wal` file, read
/// stat-only (wal_guard's lock rule), rendered here so every block spells that one fact the same
/// way. `None` renders the explicit not-obtainable line instead of dropping the field.
pub(crate) fn artifact_state_line(wal_bytes: Option<u64>) -> String {
    match wal_bytes {
        Some(bytes) => format!("artifact state: wal_size={bytes}"),
        None => ARTIFACT_STATE_UNAVAILABLE.to_string(),
    }
}

pub(crate) const ENVIRONMENT_CAUSE: &str = "cause: environment — not store damage";

/// Which severe failure a block reports. The variant's header is the block's title.
#[derive(Debug)]
pub(crate) enum FailureKind {
    StartUpRefusal,
    /// A start-up step failed without a store refusal (a store bring-up error, a launch).
    StartUpFailure,
    /// A store's periodic checkpoint failure window was exhausted — every round in it failed.
    CheckpointFailure,
    ExitCheckpointFailure,
    RuntimeIntegrityFailure,
    /// A reclaiming checkpoint was refused because the store's own page count would not shrink.
    ShrinkRefused,
}

impl FailureKind {
    /// The block's header prefix; [`FailureReport::render`] appends the rest.
    fn header(&self) -> &'static str {
        match self {
            Self::StartUpRefusal => "MahBot start-up refusal — ",
            Self::StartUpFailure => "MahBot start-up failure — ",
            Self::CheckpointFailure => "MahBot checkpoint failure — ",
            Self::ExitCheckpointFailure => "MahBot exit checkpoint failure — ",
            Self::RuntimeIntegrityFailure => "MahBot runtime integrity failure — ",
            Self::ShrinkRefused => "MahBot store shrink refused — ",
        }
    }
}

#[derive(Debug)]
pub(crate) struct FailureReport {
    kind: FailureKind,
    store: Option<&'static str>,
    db_path: Option<PathBuf>,
    /// The failing step, for a block that has no store to name (the pre-boot and launch kinds).
    step: Option<String>,
    reason: Option<String>,
    environment: bool,
    extra: Vec<String>,
}

impl FailureReport {
    #[must_use]
    pub(crate) fn new(kind: FailureKind) -> Self {
        Self {
            kind,
            store: None,
            db_path: None,
            step: None,
            reason: None,
            environment: false,
            extra: Vec::new(),
        }
    }

    #[must_use]
    pub(crate) fn store(mut self, store: &'static str) -> Self {
        self.store = Some(store);
        self
    }

    #[must_use]
    pub(crate) fn db_path(mut self, db_path: PathBuf) -> Self {
        self.db_path = Some(db_path);
        self
    }

    #[must_use]
    pub(crate) fn step(mut self, step: &str) -> Self {
        self.step = Some(step.to_string());
        self
    }

    #[must_use]
    pub(crate) fn reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    #[must_use]
    pub(crate) fn environment(mut self, environment: bool) -> Self {
        self.environment = environment;
        self
    }

    /// Append a verbatim line (e.g. a probe outcome or the stat-only artifact line).
    #[must_use]
    pub(crate) fn extra(mut self, line: impl Into<String>) -> Self {
        self.extra.push(line.into());
        self
    }

    #[must_use]
    pub(crate) fn render(&self) -> String {
        use std::fmt::Write;
        let mut body = String::new();
        let _ = writeln!(
            body,
            "{}{}",
            self.kind.header(),
            chrono::Utc::now().to_rfc3339()
        );
        match self.store {
            Some(store) => {
                let _ = writeln!(body, "store: {store}");
            }
            None => {
                let _ = writeln!(body, "{NO_STORE}");
            }
        }
        match &self.db_path {
            Some(db_path) => {
                let _ = writeln!(body, "db path: {}", db_path.display());
            }
            None => {
                let _ = writeln!(body, "{NO_DB_PATH}");
            }
        }
        if let Some(step) = &self.step {
            let _ = writeln!(body, "step: {step}");
        }
        if self.environment {
            let _ = writeln!(body, "{ENVIRONMENT_CAUSE}");
        }
        if let Some(reason) = &self.reason {
            let _ = writeln!(body, "reason: {reason}");
        }
        for line in &self.extra {
            let _ = writeln!(body, "{line}");
        }
        body
    }
}

/// One process-lifetime per-key round counter behind the "file the durable block once" rule.
pub(crate) struct RoundCounter(LazyLock<Mutex<HashMap<String, u64>>>);

impl RoundCounter {
    pub(crate) const fn new() -> Self {
        Self(LazyLock::new(|| Mutex::new(HashMap::new())))
    }

    /// Count a round for `key`: `0` on the key's first round in this process, then 1, 2, …
    pub(crate) fn prior_rounds(&self, key: &str) -> u64 {
        let mut rounds = self.0.lock().unwrap_poison();
        let count = rounds.entry(key.to_string()).or_insert(0);
        *count += 1;
        *count - 1
    }
}

/// A block's condition: the lines that say what the block is about rather than what it says.
fn condition_lines(block: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for line in block.lines() {
        if line.starts_with("MahBot ") {
            // The header: the one field two occurrences of a condition never share.
            let head = match line.rsplit_once(' ') {
                Some((head, stamp)) if stamp.contains('T') => head,
                _ => line,
            };
            kept.push(head);
        } else if line.starts_with("store: ")
            || line.starts_with("db path: ")
            || line.starts_with("step: ")
            || line.starts_with("cause: ")
        {
            kept.push(line);
        }
    }
    kept.join("\n")
}

fn condition_is_recorded(path: &Path, block: &str) -> bool {
    let Ok(existing) = std::fs::read_to_string(path) else {
        return false;
    };
    let condition = condition_lines(block);
    existing
        .split("\n\n")
        .any(|held| condition_lines(held) == condition)
}

/// Raw append: `report` as a block to `<root>/error.log`, creating the root when needed.
fn append_failure_record(root: &Path, report: &str) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    // A launch failure can be the first thing that ever needs the root: the launch creates it.
    std::fs::create_dir_all(root)?;
    let path = root.join(ERROR_LOG_NAME);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(format!("{report}\n").as_bytes())?;
    Ok(path)
}

/// Where one [`record`] call left the caller's block: the file it is in, and whether it was new.
#[derive(Debug)]
pub(crate) enum Filed {
    Appended(PathBuf),
    Held(PathBuf),
}

/// The single write entry point: file `block` in `<root>/error.log` unless the record already
/// holds a block for its condition, else report that it was already held. Never panics — a
/// recorder that could fail its caller would hide the very failure it records.
pub(crate) fn record(root: Option<&Path>, block: &str) -> Option<Filed> {
    let Some(root) = root else {
        crate::boot::timestamped_stderr(&unfiled_block(
            block,
            &format!(
                "the storage root is unresolvable — this failure could not be filed in \
                 {ERROR_LOG_NAME}"
            ),
        ));
        return None;
    };
    let path = root.join(ERROR_LOG_NAME);
    if condition_is_recorded(&path, block) {
        return Some(Filed::Held(path));
    }
    match append_failure_record(root, block) {
        Ok(path) => Some(Filed::Appended(path)),
        Err(e) => {
            crate::boot::timestamped_stderr(&unfiled_block(
                block,
                &format!(
                    "the failure could not be filed in {}: {e:#}",
                    root.join(ERROR_LOG_NAME).display(),
                ),
            ));
            None
        }
    }
}

/// The pointer an operator can follow to the record's block for this condition, or `None` when
/// the write fell back to stderr (which already carried the whole block). Every recorded kind
/// prints this one sentence on the channel it has, so the record is reachable from both. A
/// condition the record already held says so rather than reading as this launch's own filing.
pub(crate) fn recorded_pointer(what: &str, filed: Option<Filed>) -> Option<String> {
    filed.map(|filed| {
        let (verb, path) = match filed {
            Filed::Appended(path) => ("recorded", path),
            Filed::Held(path) => ("already recorded", path),
        };
        format!("{what} {verb} in {}", path.display())
    })
}

/// File `report` and point the operator at the file it landed in, on the log
/// channel (the boot path prints the same pointer on its diagnostics channel
/// instead — see [`crate::boot`]). `what` names the failure in that pointer.
pub(crate) fn record_and_point(root: Option<&Path>, what: &str, report: &FailureReport) {
    let filed = record(root, &report.render());
    if let Some(pointer) = recorded_pointer(what, filed) {
        tracing::info!("{pointer}");
    }
}

/// The stderr fallback's text: the whole block plus why it could not be filed —
/// the operator gets the failure itself, never a pointer to a file that does not
/// have it. Separated out because the record writer's only assertion surface on
/// this path is its text (libtest captures stderr).
fn unfiled_block(block: &str, reason: &str) -> String {
    format!("{block}\nnote: {reason}")
}

/// The one start-up block shape, refusal and non-refusal alike: the store and
/// its db path when the failing step was a store (a global init or a config load
/// has neither), the reason, and the environment note — so neither kind can
/// drift from the other.
pub(crate) fn start_up_report(
    kind: FailureKind,
    store: Option<(&'static str, PathBuf)>,
    reason: String,
    environment: bool,
) -> FailureReport {
    let report = FailureReport::new(kind)
        .reason(reason)
        .environment(environment);
    match store {
        Some((name, db_path)) => report.store(name).db_path(db_path),
        None => report,
    }
}

/// Record a start-up refusal block (header, store, db path, an optional
/// environment-cause line and the reason). Returns the file it was filed in, or
/// `None` when it went to stderr instead. Filed once per condition ([`record`]).
pub(crate) fn record_startup_refusal(
    root: &Path,
    store: &'static str,
    db_path: &Path,
    reason: &str,
    environment: bool,
) -> Option<Filed> {
    let report = start_up_report(
        FailureKind::StartUpRefusal,
        Some((store, db_path.to_path_buf())),
        reason.to_string(),
        environment,
    );
    record(Some(root), &report.render())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A condition that repeats across processes is filed once: the record itself
    /// carries it, which is what bounds a failure that keeps coming back launch
    /// after launch. The condition is the block's identity lines, so the same kind,
    /// store/step and cause class worded differently — its reason or an extra line
    /// included — is one block, while a different store, db path, step or cause
    /// class is a condition of its own.
    #[test]
    fn a_repeated_launch_failure_is_filed_once() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let blocks = || {
            std::fs::read_to_string(tmp.path().join(ERROR_LOG_NAME))
                .unwrap_or_default()
                .matches("MahBot start-up failure")
                .count()
        };
        // `record` answers `Some(path)` for a block it filed and for one it already
        // held (it points at the file either way), so whether a report became a
        // block is read off the file.
        let filed = |report: FailureReport| {
            let before = blocks();
            assert!(
                record(Some(tmp.path()), &report.render()).is_some(),
                "the write must land in the record"
            );
            blocks() > before
        };
        let startup = |reason: &str, environment: bool| {
            start_up_report(
                FailureKind::StartUpFailure,
                None,
                reason.to_string(),
                environment,
            )
            .step("providers::init_global")
        };

        assert!(
            filed(startup("no provider credential", false)),
            "the first failure is filed"
        );
        assert!(
            !filed(startup("the provider credential is missing", false)),
            "the same condition worded differently is not filed again"
        );
        // The one sentence the operator reads must not read as this launch's own
        // filing on every launch that repeats the condition.
        let held = recorded_pointer(
            "start-up failure",
            record(
                Some(tmp.path()),
                &startup("no provider credential", false).render(),
            ),
        );
        assert!(
            held.as_deref()
                .is_some_and(|p| p.starts_with("start-up failure already recorded in ")),
            "a condition the record holds must be pointed at as such: {held:?}"
        );
        assert!(
            !filed(
                startup("the provider credential is missing", false)
                    .extra("artifact state: wal_size=9")
            ),
            "an extra line is not identity either"
        );
        assert!(
            filed(
                startup("the provider credential is missing", false).step("config::load_or_init")
            ),
            "a different step is a condition of its own"
        );
        assert!(
            filed(startup("no provider credential", true)),
            "a different cause class is a condition of its own"
        );
        assert!(
            filed(
                FailureReport::new(FailureKind::StartUpFailure)
                    .store("core")
                    .db_path(PathBuf::from("/tmp/db/core.db"))
                    .reason("no provider credential")
            ),
            "a different artifact is a condition of its own"
        );
        assert!(
            filed(
                FailureReport::new(FailureKind::StartUpFailure)
                    .store("core")
                    .db_path(PathBuf::from("/tmp/db/logs.db"))
                    .reason("no provider credential")
            ),
            "another db path of one store is a condition of its own"
        );
        let body = std::fs::read_to_string(tmp.path().join(ERROR_LOG_NAME)).expect("error.log");
        assert_eq!(blocks(), 5, "five conditions, five blocks: {body}");
    }

    /// A root the process has not created yet still gets the block: a launch
    /// failure can be the first thing that ever needs the rest, so `record`
    /// creates the directory rather than falling back to stderr (the fallback a
    /// launch given no standard streams loses).
    #[test]
    fn record_creates_a_missing_root_directory() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let root = tmp.path().join("not-created-yet");
        let filed = record(
            Some(&root),
            &format!("MahBot start-up failure — probe\n{NO_STORE}\n"),
        );
        assert!(
            matches!(filed, Some(Filed::Appended(ref path)) if path == &root.join(ERROR_LOG_NAME)),
            "the block must be appended under the missing root: {filed:?}"
        );
        let body = std::fs::read_to_string(root.join(ERROR_LOG_NAME)).expect("error.log");
        assert!(
            body.contains("MahBot start-up failure — probe"),
            "got: {body:?}"
        );
    }
}
