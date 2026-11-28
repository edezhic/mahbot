//! What became of one piece of work handed to an agent — the one vocabulary, shared by every
//! record that answers that question.
//!
//! A job's row, an agent slot's row, a pipeline finalize, a workspace discovery pass and an
//! analysis round all ask the same question about the same kind of thing, so they answer it in the
//! same words. The stored tokens are the ones `jobs.status` and `agents.status` already carry
//! (`launched`, `done`, `done_no_answer`, `failed`), so the four that existed before this module
//! need no data conversion.
//!
//! What this module is not: a payload. It says what became of the work, never what the work
//! produced — the answer, the findings and the reason live beside it, on the row or in a struct
//! that carries one of these. `agents.outcome` is such a payload (a serialized verdict), not a
//! second vocabulary.
//!
//! The variants the product genuinely distinguishes are these seven, and no more: work still
//! running; work that finished with an answer; work that finished with nothing to report; work a
//! deliberate stop cut short; work that did not run and a later pass retries; work another writer
//! already did; and work that went wrong. A caller that needs a finer distinction carries it in
//! the payload beside the outcome, never in a new variant.

/// What became of one piece of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Accepted and not finished — the stored token `launched`, which every row already carries.
    Running,
    /// Finished and produced its answer.
    Done,
    /// Finished with nothing to report — a normal ending, not a failure.
    Empty,
    /// A deliberate stop cut it short: a workspace freeze, a cancel, a drain, a shutdown. Nothing
    /// went wrong, so it must never read as a failure.
    Stopped,
    /// It did not run, and a later pass retries it.
    Deferred,
    /// Another writer already did it — the work moved on under someone else's hand.
    Superseded,
    /// Something went wrong.
    Failed,
}

impl Outcome {
    /// The stored token: the value `jobs.status` and `agents.status` hold. `Running` is `launched`
    /// because that is the word the schema has always used.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "launched",
            Self::Done => "done",
            Self::Empty => "done_no_answer",
            Self::Stopped => "stopped",
            Self::Deferred => "deferred",
            Self::Superseded => "superseded",
            Self::Failed => "failed",
        }
    }

    /// The outcome a stored token names, or `None` for a value this build does not know. Absence
    /// of an outcome is not an outcome: the caller decides what an unknown token means.
    #[must_use]
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        match raw {
            "launched" => Some(Self::Running),
            "done" => Some(Self::Done),
            "done_no_answer" => Some(Self::Empty),
            "stopped" => Some(Self::Stopped),
            "deferred" => Some(Self::Deferred),
            "superseded" => Some(Self::Superseded),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every outcome's stored token parses back to itself, and the four that predate this module
    /// keep the tokens already stored in `jobs` and `agents`.
    #[test]
    fn every_outcome_round_trips_and_the_stored_tokens_are_unchanged() {
        assert_eq!(Outcome::Running.as_str(), "launched");
        assert_eq!(Outcome::Done.as_str(), "done");
        assert_eq!(Outcome::Empty.as_str(), "done_no_answer");
        assert_eq!(Outcome::Failed.as_str(), "failed");
        for outcome in [
            Outcome::Running,
            Outcome::Done,
            Outcome::Empty,
            Outcome::Stopped,
            Outcome::Deferred,
            Outcome::Superseded,
            Outcome::Failed,
        ] {
            assert_eq!(Outcome::parse(outcome.as_str()), Some(outcome));
            assert_eq!(outcome.to_string(), outcome.as_str());
        }
        assert_eq!(Outcome::parse("running"), None, "not a stored token");
    }
}
