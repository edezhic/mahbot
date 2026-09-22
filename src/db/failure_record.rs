//! The service's ONE durable failure record: `<root>/error.log`.
//!
//! Every severe failure the service survives — or refuses to survive — is
//! appended here as a block: a start-up refusal, any other store bring-up
//! failure, a launch failure, a persistent periodic checkpoint failure, an
//! exit-time checkpoint failure, a runtime integrity failure, and a refused
//! reclaiming-checkpoint shrink. [`record`] is the single write entry point, and
//! the append is what makes a failure durable: with a resolvable storage root the
//! block lands in the file, and without one (or when the write itself fails) the
//! full block goes to stderr instead — a fallback a launch given no standard
//! streams cannot observe (see [`crate::boot`]), so an unfiled block there is
//! silent.
//!
//! Callers render a [`FailureReport`] and hand it to [`record`]; the raw append
//! lives in [`append_failure_record`]. All of it is here so there is one file,
//! one format, and one place that knows how to append a block without tearing.
//!
//! That one entry point is bounded: [`record`] appends the block only when the
//! record does not already hold one for the same CONDITION — the block's identity
//! lines ([`condition_lines`]), what it is about rather than what it says. Two
//! occurrences of one condition worded differently (a size embedded in the reason,
//! a localised engine or OS message) are therefore one block — and so is a second,
//! genuinely different failure of the same kind on the same artifact, the price of
//! a bound that does not hinge on wording — while a different store, step or cause
//! class still files a block of its own. Because the bound lives in the writer, no
//! caller can bypass it. It covers every block this record holds — the start-up
//! refusal, the store bring-up failure, the launch failure, the checkpoint and
//! exit-checkpoint failures, the runtime integrity failure and the shrink refusal —
//! not only the repeated launch refusal it was first introduced for.
//!
//! This is the durable half of the rule the repeating in-process kinds follow
//! ([`RoundCounter`]: file the first round, then only count). A launch that exits
//! has no process to count rounds in, so the record itself is the count, keyed on
//! the kind and on the artifact or step the block names rather than on the
//! reason's wording; the in-process counter still bounds the warnings within one
//! process.
//!
//! One stated limit of this format: the engine's own reason for a failed
//! checkpoint ([`crate::db::checkpoint_cause`]) comes from a genuinely failing
//! engine checkpoint, which is not reproducible hermetically — one serialised
//! connection per store, and no fault injection into the engine — so the tests
//! that pin a checkpoint failure's block drive the cause-capturing path rather
//! than an end-to-end engine failure. The path in production is the same one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use crate::util::UnwrapPoison;

/// The record's file name under a storage root.
const ERROR_LOG_NAME: &str = "error.log";

/// The two artifact lines a block that names no store renders: a config load, a
/// provider or another global init has no store and no db path to name, and the
/// block says so rather than omitting either line — a claim about what this block
/// carries, never about the cause, since a panic inside a store bring-up reaches
/// this shape too (the binary's `bootstrap` step). Every kind whose failure IS
/// about a store fills both fields ([`FailureReport::store`] /
/// [`FailureReport::db_path`]).
pub(crate) const NO_STORE: &str = "store: none (this block names no store)";
pub(crate) const NO_DB_PATH: &str = "db path: none (this block names neither)";

/// The artifact-state line's not-obtainable wording (see [`artifact_state_line`]).
/// Kept neutral about who failed to measure: the callers that reach it are a
/// failed stat and a probe that answered nothing at all.
const ARTIFACT_STATE_UNAVAILABLE: &str = "artifact state: not obtainable (no -wal size measured)";

/// The artifact state every per-store block reports: the size of the store's
/// `-wal` file, read stat-only (wal_guard's lock rule), rendered here so every
/// block spells that one fact the same way. `None` renders the explicit
/// not-obtainable line instead of dropping the field.
pub(crate) fn artifact_state_line(wal_bytes: Option<u64>) -> String {
    match wal_bytes {
        Some(bytes) => format!("artifact state: wal_size={bytes}"),
        None => ARTIFACT_STATE_UNAVAILABLE.to_string(),
    }
}

/// The environment-cause marker: this failure is an external condition
/// (permissions, resource exhaustion, lock contention), not evidence about a
/// store's contents.
pub(crate) const ENVIRONMENT_CAUSE: &str = "cause: environment — not store damage";

/// Which severe failure a block reports. The variant's header is the block's
/// first line (followed by an RFC 3339 UTC timestamp) and is the stable identity
/// incident review and tests match on.
#[derive(Debug)]
pub(crate) enum FailureKind {
    /// A store exists but is not usable — the service refuses to start.
    StartUpRefusal,
    /// A start-up step failed without a store refusal (a store bring-up error,
    /// config, a provider or another global init).
    StartUpFailure,
    /// A store's periodic checkpoint failure window was exhausted — every round
    /// in it ended with a genuine failure and no attempt completed a checkpoint,
    /// across at least the window's rounds and span — so that round stops the
    /// service (see [`crate::db::checkpoint`]). The failing rounds before the
    /// terminal one warn only; this block is filed once, when the stop is decided.
    CheckpointFailure,
    /// A checkpoint failed on the exit-time round.
    ExitCheckpointFailure,
    /// A store's periodic integrity check failed.
    RuntimeIntegrityFailure,
    /// A reclaiming checkpoint was refused because the store's own page count
    /// did not match its files (see [`crate::db::shrink_gate`]). Not a
    /// failure: the service keeps serving.
    ShrinkRefused,
}

impl FailureKind {
    /// The block's header prefix; [`FailureReport::render`] appends the
    /// timestamp.
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

/// One block of the durable failure record, built field by field.
#[derive(Debug)]
pub(crate) struct FailureReport {
    kind: FailureKind,
    store: Option<&'static str>,
    db_path: Option<PathBuf>,
    /// The failing step, for a block that has no store to name (the pre-boot and
    /// start-up steps). Rendered after the db-path line, before the cause.
    step: Option<String>,
    reason: Option<String>,
    environment: bool,
    /// Verbatim lines appended after the fixed fields.
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

    /// Name the failing step for a block that has no store to name (a start-up or
    /// pre-boot step). Rendered as its own `step:` line — identity, per
    /// [`condition_lines`] — so the step is never merely a prefix of the reason,
    /// which two occurrences of one condition may word differently.
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

    /// Append a verbatim line (e.g. a probe outcome or the stat-only artifact
    /// state) after the fixed fields.
    #[must_use]
    pub(crate) fn extra(mut self, line: impl Into<String>) -> Self {
        self.extra.push(line.into());
        self
    }

    /// Render the block, terminated by a newline like every block in the file.
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

/// One process-lifetime per-key round counter behind the "file the durable block
/// on the first round, then only count" rule a repeating-but-survivable condition
/// follows (a persistent condition must not append a block every round). One
/// static per failure kind keeps their counts independent; the key names the
/// artifact the condition is about (a store's db path for the shrink gate, the
/// store's name for the runtime integrity check), so two different files never
/// share a count. [`crate::db::checkpoint`]'s module header owns which kinds
/// deliberately skip it and why.
pub(crate) struct RoundCounter(LazyLock<Mutex<HashMap<String, u64>>>);

impl RoundCounter {
    pub(crate) const fn new() -> Self {
        Self(LazyLock::new(|| Mutex::new(HashMap::new())))
    }

    /// Count a round for `key`: `0` on the key's first round in this process
    /// (file the durable block), otherwise the number of rounds already counted
    /// (only warn with the running count).
    pub(crate) fn prior_rounds(&self, key: &str) -> u64 {
        let mut rounds = self.0.lock().unwrap_poison();
        let count = rounds.entry(key.to_string()).or_insert(0);
        *count += 1;
        *count - 1
    }
}

/// A block's condition: the lines that say what the block is about rather than
/// what it says — its header with the trailing timestamp dropped, then the
/// `store:`, `db path:`, optional `step:` and cause lines, in that order. The
/// reason line and the extra lines are deliberately none of it: two occurrences of
/// one condition may word them differently (a size in the reason, a localised
/// engine or OS message), and [`record`] must still file the condition once.
fn condition_lines(block: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for line in block.lines() {
        if line.starts_with("MahBot ") {
            // The header: the one field two occurrences of a condition never
            // share, so the timestamp is dropped from the identity.
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

/// Whether the record at `path` provably holds a block for `block`'s condition —
/// the presence check [`record`] bounds by. The file's blocks are separated by the
/// blank-line terminator [`append_failure_record`] writes, and each is reduced to
/// its [`condition_lines`] before the comparison, so nothing else about a block
/// has to match. A file that cannot be read answers `false`: a failure is filed
/// unless the record provably holds it already.
fn condition_is_recorded(path: &Path, block: &str) -> bool {
    let Ok(existing) = std::fs::read_to_string(path) else {
        return false;
    };
    let condition = condition_lines(block);
    existing
        .split("\n\n")
        .any(|held| condition_lines(held) == condition)
}

/// Raw append: `report` as a block to `<root>/error.log`, creating the root
/// directory and the file if absent. Returns the log path. Pure `std::fs` — no
/// async, never panics on the caller's behalf. The report + terminator go out as
/// a single `write_all` (one O_APPEND write in practice); even if the libc layer
/// splits a large buffer, each chunk is offset-atomic, so the worst case under
/// concurrent store failures is interleaved chunks, never torn bytes.
fn append_failure_record(root: &Path, report: &str) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    // A launch failure can be the first thing that ever needs the root: the launch
    // path records a `temp::init_temp_root` failure against a root resolved before
    // config or self-update have created it. The record is the durable half of the
    // policy, so it creates the directory rather than losing the record — the
    // stderr fallback a launch given no standard streams loses (see `crate::boot`)
    // is exactly what the file exists to avoid.
    std::fs::create_dir_all(root)?;
    let path = root.join(ERROR_LOG_NAME);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(format!("{report}\n").as_bytes())?;
    Ok(path)
}

/// Where one [`record`] call left the caller's block: the file it is in, and
/// whether this call is what put it there. The two are different facts to the
/// operator — a condition the record already holds is deliberately not filed a
/// second time — so the pointer each renders differs.
#[derive(Debug)]
pub(crate) enum Filed {
    /// This call appended the block.
    Appended(PathBuf),
    /// The record already held a block for this condition; nothing was appended.
    Held(PathBuf),
}

/// The single write entry point: file `block` in `<root>/error.log` unless the
/// record already holds a block for the same condition, and never let the failure
/// go unrecorded. [`Filed`] names the file the block is in — appended now or
/// already there, so the caller's pointer is true either way.
///
/// The bound lives here, not in a caller, so no writer can append a second block
/// for a condition the record already carries: whether it does is the block's
/// [`condition_lines`], checked over the record by [`condition_is_recorded`]. An
/// unresolvable root or a failed write puts the FULL block on stderr instead, with
/// the reason it could not be filed, and returns `None` — the fallback the file
/// append exists to make unnecessary, and one a launch given no standard streams
/// cannot observe (see [`crate::boot`]). Never panics — a
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

/// The pointer an operator can follow to the record's block for this condition, or
/// `None` when the write fell back to stderr — the fallback already carried the
/// whole block, so there is nothing to point at. Every recorded kind prints this
/// one sentence on the channel it has (the boot diagnostics channel before tracing
/// is up, the log after), so the record is reachable from both. A condition the
/// record already held says so: the same sentence would read as this launch's own
/// filing on every launch that repeats it.
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

    /// The renderer emits the header, store, db path, step, cause, reason and
    /// extras in that order; a field that could not be obtained is rendered
    /// explicitly instead of being dropped.
    #[test]
    fn render_emits_every_field_in_order() {
        let report = FailureReport::new(FailureKind::RuntimeIntegrityFailure)
            .store("core")
            .db_path(PathBuf::from("/tmp/core.db"))
            .step("boot::open_stores")
            .environment(true)
            .reason("the check could not run")
            .extra("artifact state: wal_size=0")
            .render();
        let lines: Vec<&str> = report.lines().collect();
        let (header, timestamp) = lines[0]
            .split_once(" — ")
            .expect("the block's first line must be a header and an em-dashed timestamp");
        assert_eq!(
            header, "MahBot runtime integrity failure",
            "the header must name the failure kind: {report}"
        );
        assert!(
            !timestamp.is_empty() && timestamp.starts_with("20"),
            "the header must carry an RFC 3339 UTC timestamp: {report}"
        );
        assert_eq!(lines[1], "store: core", "got: {report}");
        assert_eq!(lines[2], "db path: /tmp/core.db", "got: {report}");
        assert_eq!(
            lines[3], "step: boot::open_stores",
            "the failing step must follow the db path: {report}"
        );
        assert_eq!(
            lines[4], ENVIRONMENT_CAUSE,
            "an environment-caused failure must say so: {report}"
        );
        assert_eq!(lines[5], "reason: the check could not run", "got: {report}");
        assert_eq!(
            lines[6], "artifact state: wal_size=0",
            "extras must follow verbatim: {report}"
        );

        let bare = FailureReport::new(FailureKind::StartUpFailure).render();
        assert!(
            bare.contains(NO_STORE) && bare.contains(NO_DB_PATH),
            "a block that names no store must say so, never omit the lines: {bare}"
        );
        assert!(
            !bare.contains(ENVIRONMENT_CAUSE)
                && !bare.contains("reason:")
                && !bare.contains("step:"),
            "absent optional fields must not be rendered: {bare}"
        );
    }

    /// A refusal block keeps the file-format contract: header, store, db path,
    /// optional cause, reason, and a trailing newline.
    #[test]
    fn startup_refusal_block_is_written_with_its_environment_note() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let db_path = tmp.path().join("db/core.db");
        record_startup_refusal(tmp.path(), "core", &db_path, "permission denied", true);
        let body = std::fs::read_to_string(tmp.path().join(ERROR_LOG_NAME)).expect("error.log");
        assert!(
            body.starts_with("MahBot start-up refusal — ")
                && body.contains("store: core\n")
                && body.contains(&format!("db path: {}\n", db_path.display()))
                && body.contains(&format!("{ENVIRONMENT_CAUSE}\n"))
                && body.contains("reason: permission denied\n"),
            "got: {body}"
        );
    }

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

    /// The block is written to the file under the resolved root, with the file's
    /// blank-line terminator. The stderr fallback for an unresolved or unwritable
    /// root is asserted in text by [`unfiled_block_carries_the_block_and_the_reason`]
    /// (libtest owns stderr, so it cannot be captured here).
    #[test]
    fn record_files_the_block_with_the_file_terminator() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let block = format!("MahBot start-up failure — probe\n{NO_STORE}\n");
        record(Some(tmp.path()), &block);
        let body = std::fs::read_to_string(tmp.path().join(ERROR_LOG_NAME)).expect("error.log");
        assert_eq!(
            body,
            format!("{block}\n"),
            "the block and the file's blank-line terminator must be written: {body:?}"
        );
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

    /// The stderr fallback in text: a block that cannot be filed reaches stderr
    /// whole, followed by the reason it could not be filed — never a pointer to
    /// a record that does not hold it.
    #[test]
    fn unfiled_block_carries_the_block_and_the_reason() {
        let text = unfiled_block(
            "MahBot start-up failure — probe\nstore: core\n",
            "the storage root is unresolvable",
        );
        assert!(
            text.starts_with("MahBot start-up failure — probe\nstore: core\n"),
            "the whole block must reach stderr: {text:?}"
        );
        assert!(
            text.contains("note: the storage root is unresolvable"),
            "the fallback must say why it is on stderr: {text:?}"
        );
    }
}
