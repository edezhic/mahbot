//! How long an inspection read may take before the command that started it ends.
//!
//! The read-only clients read stores the product may not be serving them from:
//! this process opens the store directly, or asks the instance over the debug
//! channel. Either read can fail to return, and one that never returns holds the
//! command — and the store file it opened — for as long as it runs. So the
//! command carries its own answer for how long it may wait.
//!
//! The limit is the COMMAND's, not one statement's or one wait's: it starts when
//! the command starts reading, and a spent budget fails the read it was spent on
//! and every read after it instead of being reset per statement, per wait or per
//! store. The smallest value is an immediate failure, never an unbounded read —
//! the engine's own "zero means no limit" applies to the engine's limit, not to
//! this one, which the engine never sees.
//!
//! Two mechanisms enforce it, and the second exists because the first cannot
//! always run:
//! - [`Read::check`] is the cooperative half: a read asks the budget whether it
//!   is spent before each step, before each statement of a multi-statement walk,
//!   and once the read is over. A spent budget is the limit's failure wherever it
//!   is asked — including at the end, so a read that ran past the limit is not an
//!   answer even though it came back.
//! - [`Watchdog`] is the half that does not depend on the read yielding: it ends
//!   the process past the limit, which is what covers a read that never comes back
//!   to its own check. Its grace ([`WATCHDOG_GRACE_SECS`]) is what keeps the
//!   reporting with a read that does come back inside it; one that comes back
//!   later than the grace is cut by the process ending instead. The route that
//!   makes the reads arms it ([`Read::arm`]) — once for the whole route, not per
//!   attempt or per round trip.
//!
//! This limit is the client's own; the engine's per-statement guard
//! ([`crate::db::store::statement_limit`]) is separate and stays where it is. A
//! read limit raised ABOVE that guard therefore lets the guard answer first, and
//! its own wording is what the user sees — passed through unchanged, never
//! rewritten as this limit's failure. The default sits below the guard, so the
//! default behaviour is this limit's.

use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::anyhow;

/// Env override for the limit, in seconds — the same whole-seconds override the
/// product's other bounded waits use ([`crate::util::env_duration_secs`]), read
/// at each call so a shortened limit is a deterministic outcome for a check.
pub(crate) const ENV: &str = "MAHBOT_DEBUG_READ_LIMIT_SECS";

/// Default limit. Well past an ordinary read of the largest store the product
/// keeps: measured here on a purpose-built store of the product's own log schema,
/// built larger than the biggest real one — 814 MB and 1.5M rows against the real
/// `core.db`'s ~667 MB — a whole-schema dump takes ~1.4 s with the page cache warm
/// and ~1.7 s with it evicted, about a fortieth of the limit (and that is the
/// DEBUG build, the slower one). Well short of the per-statement limit the engine
/// enforces on its own connections ([`crate::db::store::statement_limit`]) so that
/// on a slow read THIS limit is the one that ends the command and its failure is
/// the one the user sees.
const DEFAULT_SECS: u64 = 60;

/// Upper bound on the override: a day is past any read worth waiting for, and it
/// keeps an absurd knob value — `Instant + Duration` panics on overflow — from
/// turning the budget's own deadline and cutoff into a panic.
const MAX_SECS: u64 = 86_400;

/// How far past the limit the watchdog waits before ending the process.
///
/// This is the cooperative half's own margin, and it is small: a read that yields
/// (each step, each round trip) notices the spent budget as soon as it returns, so
/// the watchdog is here for the read that never returns to its loop — and the
/// closer it fires to the limit, the closer the command ends to the limit it
/// promised.
pub(crate) const WATCHDOG_GRACE_SECS: u64 = 1;

/// What the process the watchdog ends exits with: the command's error output and
/// its exit status are its only surface — it opens no log (an instance holds the
/// stores) and has nothing durable to write to — and `1` is the failure status its
/// other failures exit with.
pub(crate) const WATCHDOG_EXIT_CODE: i32 = 1;

/// The command's read limit.
fn limit() -> Duration {
    crate::util::env_duration_secs(ENV, DEFAULT_SECS).min(Duration::from_secs(MAX_SECS))
}

/// What a read the limit ended says: the target the user asked about, the limit it
/// did not finish within, and what the read was left in. The wording holds for
/// every way the limit can end a read — one cut mid-flight, one that came back
/// past the limit, and one that was never attempted because the budget was already
/// spent. The instance route's own half says the instance is holding the store and
/// that its file was left untouched — which is what separates this failure from a
/// channel that is not answering at all, and from a store that was opened behind
/// the instance's back.
#[must_use]
fn limit_message(target: &str, limit: Duration, over_channel: bool) -> String {
    let state = if over_channel {
        "the instance holds the store and its file was left untouched; raise the limit for a \
         slower read, or re-run once the instance is idle"
    } else {
        "raise the limit for a slower read"
    };
    format!(
        "'{target}' was not read within the {}s read limit ({ENV}) — {state}",
        limit.as_secs()
    )
}

/// The command's read budget: the one clock every read of the command runs under.
pub(crate) struct Budget {
    limit: Duration,
    deadline: Instant,
    /// [`Self::deadline`] plus [`WATCHDOG_GRACE_SECS`]: when a read that will not
    /// yield is cut by the process ending.
    cutoff: Instant,
}

impl Budget {
    /// Start the budget — the clock starts here, where the command starts reading.
    #[must_use]
    pub(crate) fn start() -> Self {
        let limit = limit();
        let deadline = Instant::now() + limit;
        Self {
            limit,
            deadline,
            cutoff: deadline + Duration::from_secs(WATCHDOG_GRACE_SECS),
        }
    }

    /// What is left of the budget; zero once it is spent.
    fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

/// One read of a command in flight: the target the user asked about (what a
/// failure names) and the budget the whole command runs under.
#[derive(Clone, Copy)]
pub(crate) struct Read<'a> {
    target: &'a str,
    budget: &'a Budget,
    over_channel: bool,
}

impl<'a> Read<'a> {
    /// A read this process makes of a store it opened itself.
    #[must_use]
    pub(crate) fn direct(target: &'a str, budget: &'a Budget) -> Self {
        Self {
            target,
            budget,
            over_channel: false,
        }
    }

    /// A read the instance answers over the debug channel. The failure still
    /// names the target the user asked about (the store, or `config`); the only
    /// thing the placement adds is that an instance holds the store.
    #[must_use]
    pub(crate) fn over_channel(target: &'a str, budget: &'a Budget) -> Self {
        Self {
            target,
            budget,
            over_channel: true,
        }
    }

    /// What is left of the command's budget, for a wait this read hands to
    /// something else (the instance's answer).
    pub(crate) fn remaining(&self) -> Duration {
        self.budget.remaining()
    }

    /// The limit's failure for this read.
    #[must_use]
    pub(crate) fn timeout(&self) -> anyhow::Error {
        anyhow!(self.message())
    }

    /// The limit's failure as the sentence the watchdog prints.
    fn message(&self) -> String {
        limit_message(self.target, self.budget.limit, self.over_channel)
    }

    /// Fail at once if the budget is already spent — the cooperative half of the
    /// limit, asked by a read that yields: before each step, before each statement
    /// of a multi-statement walk, and once the read is over. Asked at the end, it
    /// is what makes the limit's verdict final: a read that ran past the limit is
    /// the limit's failure even though it came back.
    pub(crate) fn check(&self) -> anyhow::Result<()> {
        if self.remaining().is_zero() {
            return Err(self.timeout());
        }
        Ok(())
    }

    /// [`Self::check`] plus the watchdog, for a read that is about to run and
    /// must end even if it never yields. The check comes first, so a spent
    /// budget is reported rather than armed with an already-passed cutoff that
    /// would end the process mid-report.
    pub(crate) fn arm(&self) -> anyhow::Result<Watchdog> {
        self.check()?;
        Ok(Watchdog::arm(self.budget.cutoff, self.message()))
    }
}

/// Ends the process if the read it guards is still running at its budget's
/// cutoff — the half of the limit that does not depend on the read yielding.
///
/// Disarmed by dropping it, so it must be bound to a named variable that lives as
/// long as the read does: `let _watchdog = read.arm()?;`. Every route keeps it
/// for the reading alone and drops it before the answer is written, so a read
/// that came back is printed as its answer — the process is ended only while a
/// read is still running.
///
/// In a test binary the process-ending half is off ([`enable_process_exit`]):
/// ending the process there would take the whole test run down instead of the one
/// failing test, so the path is exercised only by the child of
/// [`crate::db::debug`]'s watchdog test, where it is a process of its own.
#[must_use]
pub(crate) struct Watchdog {
    /// Held only to keep the channel alive: the guard thread wakes when this
    /// drops, and ends the process if the cutoff arrives first. `None` when
    /// nothing is armed (a test binary).
    _armed: Option<Sender<()>>,
}

/// Whether the watchdog may end this process — see [`Watchdog`]. Never set in a
/// test binary except by the child of the debug watchdog test.
#[cfg(test)]
static PROCESS_EXIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Let this process end itself when the read limit is spent. Called only by the
/// child process of the debug watchdog test: "the process is gone" is observable
/// across two processes, and a watchdog firing inside the test binary would end
/// the whole run.
#[cfg(test)]
pub(crate) fn enable_process_exit() {
    PROCESS_EXIT.store(true, std::sync::atomic::Ordering::Relaxed);
}

impl Watchdog {
    fn arm(cutoff: Instant, message: String) -> Self {
        #[cfg(test)]
        if !PROCESS_EXIT.load(std::sync::atomic::Ordering::Relaxed) {
            return Self { _armed: None };
        }
        let (tx, rx) = channel::<()>();
        std::thread::spawn(move || {
            if rx.recv_timeout(cutoff.saturating_duration_since(Instant::now()))
                == Err(RecvTimeoutError::Timeout)
            {
                // The same shape and status the read's own failure would take
                // (see [`WATCHDOG_EXIT_CODE`]).
                crate::util::print_stderr(&format!("Error: {message}"));
                std::process::exit(WATCHDOG_EXIT_CODE);
            }
        });
        Self { _armed: Some(tx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The limit one knob value yields, with the knob removed again before the
    /// next value is set (the guard holds the shared env lock, so only one may
    /// live at a time).
    fn limit_with(value: Option<&str>) -> Duration {
        let _env = crate::util::test::set_env_var(ENV, value);
        limit()
    }

    /// Unset means the default, an override is taken as given, an absurd one is
    /// clamped — and the smallest value is an immediate failure, never an
    /// engine-style "no limit".
    #[test]
    #[serial_test::serial(read_limit)]
    fn limit_reads_the_knob_within_its_bounds() {
        assert_eq!(
            limit_with(None),
            Duration::from_secs(DEFAULT_SECS),
            "an unset knob means the default"
        );
        assert_eq!(
            limit_with(Some("3")),
            Duration::from_secs(3),
            "an override is taken as given"
        );
        assert_eq!(
            limit_with(Some("99999999")),
            Duration::from_secs(MAX_SECS),
            "an absurd value is clamped"
        );
        assert_eq!(
            limit_with(Some("0")),
            Duration::ZERO,
            "zero is an immediate failure, never 'no limit'"
        );
    }
}
