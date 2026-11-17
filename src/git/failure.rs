//! What a failed git step can honestly tell the owner: the step that failed, git's own reason,
//! and which outcome the text proves.
//!
//! The classification is deliberately conservative. A remote that answered and refused is not a
//! remote that could not be reached, and a cause the product cannot tell apart (a 403 that is
//! either a missing token scope or no write access, a 404 that is either a wrong path or a
//! private repository) is never picked: the message repeats git's own words and claims nothing
//! beyond them. Credential material is scrubbed here, once, so every surface that shows or
//! stores the text — toast, tooltip, modal, logs — is covered.

use std::fmt;
use std::sync::LazyLock;

use regex::Regex;

/// The step a sync-path failure came from, so every message names what did not happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitStep {
    /// Asking the destination which commits it carries.
    Probe,
    Pull,
    Push,
    /// Resolving what a press would do, before any network step ran.
    Sync,
}

impl GitStep {
    /// The log field's spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::Pull => "pull",
            Self::Push => "push",
            Self::Sync => "sync",
        }
    }
}

/// What a failure proves about the remote. Only these claims are made; everything the text does
/// not settle stays [`Self::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitFailureKind {
    /// No answer came back at all: DNS, connect, a dropped connection, the deadline.
    Unreachable,
    /// The remote answered and refused the credentials or the access.
    AccessRefused,
    /// The remote answered and rejected the update for a reason that is not access.
    UpdateRejected,
    /// The failure happened locally: git could not start, had no credential to offer, the
    /// repository is unusable, or an update that arrived could not be applied on disk.
    Local,
    /// The text settles nothing.
    Unknown,
}

impl GitFailureKind {
    /// The log field's spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unreachable => "unreachable",
            Self::AccessRefused => "access-refused",
            Self::UpdateRejected => "update-rejected",
            Self::Local => "local",
            Self::Unknown => "unknown",
        }
    }

    /// The claim the message makes; `None` for [`Self::Unknown`], where the message is git's own
    /// words alone.
    const fn clause(self) -> Option<&'static str> {
        match self {
            Self::Unreachable => Some("the remote could not be reached"),
            Self::AccessRefused => Some("the remote refused access"),
            Self::UpdateRejected => Some("the remote rejected the update"),
            Self::Local => Some("the failure was local"),
            Self::Unknown => None,
        }
    }
}

/// How much of git's text is kept. Its cause can sit at either end of the report (`fatal: …`
/// opens a transport failure, `! [remote rejected] …` closes a refused push), so the cut keeps
/// both ends (see [`crate::util::truncate_sandwich`]) and a pathological report cannot grow the
/// log row or the modal without bound.
const MAX_REASON_BYTES: usize = 600;

/// How much of the reason the tooltip's single line keeps.
const MAX_SHORT_REASON_BYTES: usize = 160;

/// The line prefixes that are never the cause: git's own hints, its continuation lines, and the
/// `To …`/`From …` header a transport report opens with.
const CAUSE_SKIP_PREFIXES: &[&str] = &["hint:", "note:", "...", "to ", "from "];

/// A single line cut to [`MAX_SHORT_REASON_BYTES`] keeping both ends: git's cause can sit at
/// either end of a long line (`fatal: unable to access '<url>': … error: 403`).
fn cut_line(line: &str) -> String {
    if line.len() <= MAX_SHORT_REASON_BYTES {
        return line.to_string();
    }
    let head = MAX_SHORT_REASON_BYTES * 2 / 3;
    let tail = MAX_SHORT_REASON_BYTES / 3;
    format!(
        "{}\u{2026}{}",
        &line[..line.floor_char_boundary(head)],
        &line[line.ceil_char_boundary(line.len() - tail)..],
    )
}

/// One failed git step: what was attempted, what git said, and what that proves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitFailure {
    step: GitStep,
    remote: Option<String>,
    kind: GitFailureKind,
    /// git's own words — or the product's, where git said nothing usable — scrubbed and capped;
    /// empty when nothing was said.
    reason: String,
}

impl GitFailure {
    /// Build the failure from git's text, scrubbed and capped once so every surface is covered.
    #[must_use]
    pub(crate) fn new(
        step: GitStep,
        remote: Option<&str>,
        kind: GitFailureKind,
        reason: &str,
    ) -> Self {
        Self {
            step,
            remote: remote.map(str::to_string),
            kind,
            reason: crate::util::truncate_sandwich(
                &redact(reason.trim()),
                MAX_REASON_BYTES,
                "git output",
            )
            .trim()
            .to_string(),
        }
    }

    /// A failure with no remote involved — the step could not even run.
    #[must_use]
    pub(crate) fn local(step: GitStep, remote: Option<&str>, reason: &str) -> Self {
        Self::new(step, remote, GitFailureKind::Local, reason)
    }

    /// A failure from git's own text, letting the text decide what it proves.
    #[must_use]
    pub(crate) fn classified(step: GitStep, remote: Option<&str>, report: &str) -> Self {
        Self::new(step, remote, classify(report), report)
    }

    #[must_use]
    pub const fn step(&self) -> GitStep {
        self.step
    }

    #[must_use]
    pub const fn kind(&self) -> GitFailureKind {
        self.kind
    }

    #[must_use]
    pub fn remote(&self) -> Option<&str> {
        self.remote.as_deref()
    }

    /// The sentence the modal and the log row show: the step, the claim the text proves, and
    /// git's own reason in full.
    #[must_use]
    pub fn message(&self) -> String {
        self.sentence(&self.compose((!self.reason.is_empty()).then_some(self.reason.as_str())))
    }

    /// One line for the toast: the step, the claim, and the line of git's text that names the
    /// cause — the toast is gone in four seconds, while the modal and the log keep the full text.
    #[must_use]
    pub fn toast(&self) -> String {
        self.sentence(&self.short())
    }

    /// One line for the sync control's tooltip: the claim and the line of git's text that names
    /// the cause, capped at both ends. The modal and the log keep the full text.
    #[must_use]
    pub fn short(&self) -> String {
        self.compose(self.cause_line().map(cut_line).as_deref())
    }

    /// The step that failed, then `text`.
    fn sentence(&self, text: &str) -> String {
        format!(
            "Sync could not {}: {text}",
            self.action().unwrap_or_else(|| "run".to_string())
        )
    }

    /// The claim the text proves followed by `text`; a missing claim or a missing text is left
    /// out rather than filled in, and both missing is stated as such.
    fn compose(&self, text: Option<&str>) -> String {
        match (self.kind.clause(), text) {
            (Some(claim), Some(text)) => format!("{claim} \u{2014} {text}"),
            (Some(claim), None) => claim.to_string(),
            (None, Some(text)) => text.to_string(),
            (None, None) => "git gave no reason".to_string(),
        }
    }

    /// The line of git's text that names the cause: the first one that is not a header, a hint or
    /// git's trailing advice. git puts its most specific message first and its `hint:`/advice
    /// lines after the generic summary, so the first real line is the one that says why. The last
    /// non-empty line is the fallback when every line is one of those.
    fn cause_line(&self) -> Option<&str> {
        let mut first = None;
        let mut last = None;
        for line in self
            .reason
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            let skipped = CAUSE_SKIP_PREFIXES
                .iter()
                .any(|prefix| starts_with_ignore_case(line, prefix));
            if first.is_none() && !skipped {
                first = Some(line);
            }
            last = Some(line);
        }
        first.or(last)
    }

    /// The reason as a log dedupe key: the step and the cause, with git's own measurements
    /// (`after 75002 ms`) folded out because they differ between attempts at the same failure.
    /// The step is part of the key, so a probe that fails the same way does not swallow the pull
    /// or push that failed for the same reason afterwards.
    #[must_use]
    pub fn dedupe_key(&self) -> String {
        static ELAPSED: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(?i)\bafter \d+ (?:ms|milliseconds?)\b")
                .expect("the elapsed pattern compiles")
        });
        format!(
            "{}:{}",
            self.step.label(),
            ELAPSED.replace_all(&self.reason, "after ? ms")
        )
    }

    /// What the message says the product could not do; `None` for the whole-sync step.
    fn action(&self) -> Option<String> {
        match self.step {
            GitStep::Probe => Some(format!(
                "ask {} what it carries",
                self.remote.as_deref().unwrap_or("the remote")
            )),
            GitStep::Pull => Some("pull".to_string()),
            GitStep::Push => Some("push".to_string()),
            GitStep::Sync => None,
        }
    }
}

/// Whether `line` opens with `prefix`, ignoring ASCII case and without allocating.
fn starts_with_ignore_case(line: &str, prefix: &str) -> bool {
    line.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// git's report is both streams: its own refusal is on stderr, a local hook's reason on stdout.
pub(crate) fn report(stderr: &str, stdout: &str) -> String {
    [stderr.trim(), stdout.trim()]
        .into_iter()
        .filter(|stream| !stream.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Why a network git command did not answer — the raw material [`GitFailure`] is built from.
/// Kept apart from the classified failure so a caller can tell "git never started", "the deadline
/// passed" and "git exited" without reading git's text.
#[derive(Debug)]
pub(crate) enum GitNetworkFailure {
    /// git could not be started at all.
    Spawn { command: String, error: String },
    /// Waiting for git failed, so its exit state is unknown — nothing was heard from a remote.
    WaitFailed { command: String, error: String },
    /// No answer came back before the deadline.
    TimedOut { command: String, secs: u64 },
    /// git ran and exited non-zero; `stderr` and `stdout` are its own report — a local pre-push
    /// hook's reason lands on stdout, git's own refusal on stderr.
    Exited {
        command: String,
        status: Option<i32>,
        stderr: String,
        stdout: String,
    },
}

impl GitNetworkFailure {
    /// The failure to show, attributed to `step` and to the `remote` it asked (when it asked one).
    #[must_use]
    pub(crate) fn into_failure(self, step: GitStep, remote: Option<&str>) -> GitFailure {
        match self {
            Self::Spawn { error, .. } => {
                GitFailure::local(step, remote, &format!("git could not be started: {error}"))
            }
            Self::WaitFailed { command, error } => GitFailure::local(
                step,
                remote,
                &format!("git {command} could not be waited for: {error}"),
            ),
            Self::TimedOut { secs, .. } => GitFailure::new(
                step,
                remote,
                GitFailureKind::Unreachable,
                &format!("no answer came back within {secs}s"),
            ),
            Self::Exited {
                status,
                stderr,
                stdout,
                ..
            } => {
                let report = report(&stderr, &stdout);
                let reason = if report.is_empty() {
                    // git exited without saying anything: state the exit, never invent a cause.
                    status.map_or_else(String::new, |code| {
                        format!("git exited with status {code} and printed nothing")
                    })
                } else {
                    report
                };
                GitFailure::classified(step, remote, &reason)
            }
        }
    }
}

impl fmt::Display for GitNetworkFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { command, error } => {
                write!(f, "git {command} could not be started: {error}")
            }
            Self::WaitFailed { command, error } => {
                write!(f, "git {command} could not be waited for: {error}")
            }
            Self::TimedOut { command, secs } => {
                write!(f, "git {command} timed out after {secs}s")
            }
            Self::Exited {
                command,
                stderr,
                stdout,
                ..
            } => write!(
                f,
                "git {command} failed: {}",
                redact(&report(stderr, stdout))
            ),
        }
    }
}

impl std::error::Error for GitNetworkFailure {}

/// git's text that means the failure happened locally. Matched against the lines git did not
/// prefix `remote:` — a server-side `unable to create temporary file` is the destination's
/// problem, never a local one.
const LOCAL_MARKERS: &[&str] = &[
    "terminal prompts disabled",
    "could not read username",
    "could not read password",
    // git's own words for "this directory is not a work tree", in its two forms. A path git was
    // asked about that `does not appear to be a git repository` is the *destination* answering,
    // never this: a local marker means the failure was local, not that no remote was reached.
    "not a git repository:",
    "not a git repository (or any",
    "index.lock",
    "is not a git command",
    "unable to create",
    "could not create directory",
    "unable to write",
    "unable to open",
    "could not open",
    "cannot open",
    "unable to unlink",
    "could not unlink",
    "unable to reset",
    "could not reset index",
    // Only a local ref/index update: `unable to update url base from redirection` is the
    // transport answering, and must never be called a local failure.
    "unable to update ref",
    "could not update ref",
];

/// git's text that means the remote answered and refused the credentials or the access. A bare
/// "permission denied" is a refusal here only because the local write failures that print it are
/// matched above first. The broad "not permitted"/"not authorized" spellings are deliberately
/// narrowed to the access context: the same words carry a host's policy refusal, which is an
/// update rejection, and guessing access from them would claim a cause the text does not settle.
const ACCESS_MARKERS: &[&str] = &[
    "authentication failed",
    "authentication required",
    "access denied",
    "permission denied",
    "invalid username or password",
    "invalid credentials",
    "returned error: 401",
    "returned error: 403",
    "http 401",
    "http 403",
    "denied to ",
    "not allowed to push",
    "not permitted to access",
    "not authorized to access",
    "not authorized to push",
];

/// git's text that means the update was rejected by policy, never by access: a protected branch
/// or a branch policy is reported with a hook refusal and with wording that reads like an access
/// denial, so it is settled before any access marker is considered.
const POLICY_MARKERS: &[&str] = &[
    "protected branch",
    "branch is protected",
    "gh006",
    // Azure DevOps states its branch policy with wording an access marker would otherwise take.
    "pushes to this branch are not permitted",
    // Gerrit's own prohibition, in the same shape.
    "prohibited by",
];

/// git's text that means the remote answered and rejected the update for another reason.
const REJECTED_MARKERS: &[&str] = &[
    "[rejected]",
    "[remote rejected]",
    "non-fast-forward",
    "fetch first",
    "failed to push some refs",
    "hook declined",
    "pre-receive hook",
    "update declined",
    "stale info",
];

/// git's text that means no answer came back at all.
const UNREACHABLE_MARKERS: &[&str] = &[
    "could not resolve host",
    "failed to connect",
    "couldn't connect to server",
    "connection refused",
    "connection timed out",
    "operation timed out",
    "connection reset",
    "remote end hung up",
    "network is unreachable",
    "no route to host",
    "name or service not known",
    "temporary failure in name resolution",
];

/// What git's report proves about the remote.
///
/// Local markers come first, and only against the lines git did not attribute to the server (a
/// missing credential fails before anything is asked). Then a policy rejection — a protected
/// branch, a branch policy, a host's prohibition — which is an update rejection whatever wording
/// accompanies it. Then access refusals — an SSH refusal prints `Permission denied (publickey)`
/// *and* `Could not read from remote repository`, and the refusal is the answer — then the other
/// update rejections, and only then reachability, so a transport line never turns a refusal into
/// an unreachable remote.
fn classify(report: &str) -> GitFailureKind {
    let local_text = report
        .lines()
        .filter(|line| !line.trim_start().starts_with("remote:"))
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();
    let text = report.to_lowercase();
    let has =
        |haystack: &str, markers: &[&str]| markers.iter().any(|marker| haystack.contains(marker));
    if has(&local_text, LOCAL_MARKERS) {
        GitFailureKind::Local
    } else if has(&text, POLICY_MARKERS) {
        GitFailureKind::UpdateRejected
    } else if has(&text, ACCESS_MARKERS) {
        GitFailureKind::AccessRefused
    } else if has(&text, REJECTED_MARKERS) {
        GitFailureKind::UpdateRejected
    } else if has(&text, UNREACHABLE_MARKERS) {
        GitFailureKind::Unreachable
    } else {
        GitFailureKind::Unknown
    }
}

/// The userinfo of any URL git echoes back: `https://user:token@host` becomes `https://***@host`.
/// The only credential material git's own report carries is the URL it was given. A userinfo with
/// a colon may contain `/` (a password can carry one unencoded), so the colon form is matched
/// first and allowed to cross a slash; without a colon a `/` ends the authority and the `@` is
/// part of a path, never a credential.
fn redact(text: &str) -> String {
    static USERINFO: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)(https?|ssh|git|ftp)://(?:[^/\s@]*:[^\s@]*|[^/\s@]+)@")
            .expect("the credential pattern compiles")
    });
    USERINFO.replace_all(text, "${1}://***@").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the taxonomy: a refusal is never reported as an unreachable remote, a
    /// non-access rejection is never called an access denial, and a cause the text does not
    /// settle is claimed by neither.
    #[expect(clippy::too_many_lines)] // one table row per real git stderr shape
    #[test]
    fn classification_cases() {
        struct Case {
            name: &'static str,
            stderr: &'static str,
            kind: GitFailureKind,
        }
        let cases: &[Case] = &[
            Case {
                name: "https bad credentials",
                stderr: "remote: Invalid username or password.\n\
                         fatal: Authentication failed for 'https://github.com/o/r.git/'",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "https 403",
                stderr: "fatal: unable to access 'https://github.com/o/r.git/': \
                         The requested URL returned error: 403",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "ssh publickey refused",
                stderr: "git@github.com: Permission denied (publickey).\n\
                         fatal: Could not read from remote repository.",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "permission denied to the repository",
                stderr: "remote: Permission to o/r.git denied to someone.\n\
                         fatal: unable to access 'https://github.com/o/r.git/': \
                         The requested URL returned error: 403",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "no credential and prompting disabled",
                stderr: "fatal: could not read Username for 'https://github.com': \
                         terminal prompts disabled",
                kind: GitFailureKind::Local,
            },
            Case {
                name: "dns failure",
                stderr: "fatal: unable to access 'https://github.com/o/r.git/': \
                         Could not resolve host: github.com",
                kind: GitFailureKind::Unreachable,
            },
            Case {
                name: "ssh connect timeout",
                stderr: "ssh: connect to host github.com port 22: Connection timed out\n\
                         fatal: Could not read from remote repository.",
                kind: GitFailureKind::Unreachable,
            },
            Case {
                name: "diverged push",
                stderr: " ! [rejected]        main -> main (non-fast-forward)\n\
                         error: failed to push some refs to 'origin'",
                kind: GitFailureKind::UpdateRejected,
            },
            Case {
                name: "protected branch",
                stderr: "remote: error: GH006: Protected branch update failed for refs/heads/main.\n\
                         ! [remote rejected] main -> main (protected branch hook declined)",
                kind: GitFailureKind::UpdateRejected,
            },
            Case {
                name: "pre-receive hook",
                stderr: "remote: pre-receive hook declined\n\
                         ! [remote rejected] main -> main (pre-receive hook declined)",
                kind: GitFailureKind::UpdateRejected,
            },
            Case {
                name: "gerrit policy refusal",
                stderr: "remote: You are not permitted to perform this action.\n\
                         ! [remote rejected] refs/heads/main -> refs/heads/main \
                         (prohibited by Gerrit)",
                kind: GitFailureKind::UpdateRejected,
            },
            Case {
                name: "bitbucket access refusal",
                stderr: "remote: You are not authorized to access this repository.\n\
                         fatal: unable to access 'https://bitbucket.example/o/r.git/': \
                         The requested URL returned error: 401",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "a redirected url base is not a local failure",
                stderr: "fatal: unable to update url base from redirection:\n\
                         asked for: https://x/r.git/info/refs?service=git-upload-pack\n\
                         redirect: https://y/r.git/info/refs?service=git-upload-pack",
                kind: GitFailureKind::Unknown,
            },
            Case {
                name: "azure devops branch policy",
                stderr: "remote: TF402455: Pushes to this branch are not permitted; you must use a \
                         pull request.\n\
                         ! [remote rejected] main -> main (TF402455)",
                kind: GitFailureKind::UpdateRejected,
            },
            Case {
                name: "gitlab protected branch",
                stderr: "remote: GitLab: You are not allowed to push code to protected branches \
                         on this project.\n\
                         ! [remote rejected] main -> main (pre-receive hook declined)",
                kind: GitFailureKind::UpdateRejected,
            },
            Case {
                name: "gitlab no write access refused by a hook",
                stderr: "remote: GitLab: You are not allowed to push code to this project.\n\
                         ! [remote rejected] main -> main (pre-receive hook declined)\n\
                         error: failed to push some refs to 'git@gitlab.com:o/r.git'",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "gitea no write access refused by a hook",
                stderr: "remote: User permission denied for writing.\n\
                         ! [remote rejected] main -> main (pre-receive hook declined)\n\
                         error: failed to push some refs to 'git@gitea.com:o/r.git'",
                kind: GitFailureKind::AccessRefused,
            },
            Case {
                name: "local write denial during a pull",
                stderr: "error: unable to unlink old 'README.md': Permission denied\n\
                         fatal: Could not reset index file to revision 'FETCH_HEAD'.",
                kind: GitFailureKind::Local,
            },
            Case {
                name: "the destination's own disk failure is not local",
                stderr: "remote: error: unable to create temporary file: No space left on device\n\
                         error: unpack failed: unpack-objects abnormal exit",
                kind: GitFailureKind::Unknown,
            },
            Case {
                name: "dropped connection",
                stderr: "fatal: the remote end hung up unexpectedly",
                kind: GitFailureKind::Unreachable,
            },
            Case {
                name: "https 404 picks no cause",
                stderr: "fatal: unable to access 'https://github.com/o/r.git/': \
                         The requested URL returned error: 404",
                kind: GitFailureKind::Unknown,
            },
            Case {
                name: "tls failure claims nothing",
                stderr: "fatal: unable to access 'https://github.com/o/r.git/': \
                         SSL certificate problem: unable to get local issuer certificate",
                kind: GitFailureKind::Unknown,
            },
            Case {
                name: "local write denial is not an access refusal",
                stderr: "error: unable to write file src/main.rs: Permission denied",
                kind: GitFailureKind::Local,
            },
            Case {
                name: "a path that is not a repository is the destination answering",
                stderr: "fatal: '/tmp/x/origin.git' does not appear to be a git repository\n\
                         fatal: Could not read from remote repository.",
                kind: GitFailureKind::Unknown,
            },
            Case {
                name: "nothing printed",
                stderr: "",
                kind: GitFailureKind::Unknown,
            },
        ];
        for case in cases {
            assert_eq!(
                classify(case.stderr),
                case.kind,
                "case {}: {:?}",
                case.name,
                case.stderr
            );
        }
    }

    /// The message names the step, states only what the text proves, and keeps git's own reason —
    /// including the tail line where git puts the cause.
    #[test]
    fn message_keeps_the_claim_and_the_cause() {
        let refused = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "remote: Invalid username or password.",
        );
        assert_eq!(
            refused.message(),
            "Sync could not push: the remote refused access \u{2014} \
             remote: Invalid username or password."
        );

        let unreachable = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::Unreachable,
            "fatal: unable to access 'https://github.com/o/r.git/': Could not resolve host: github.com",
        );
        assert!(
            unreachable.message().starts_with(
                "Sync could not ask origin what it carries: the remote could not be reached \u{2014} "
            ),
            "{}",
            unreachable.message()
        );

        let unknown = GitFailure::new(GitStep::Pull, None, GitFailureKind::Unknown, "fatal: odd");
        assert_eq!(unknown.message(), "Sync could not pull: fatal: odd");

        let silent = GitFailure::new(GitStep::Push, None, GitFailureKind::UpdateRejected, "");
        assert_eq!(
            silent.message(),
            "Sync could not push: the remote rejected the update"
        );

        let nothing = GitFailure::new(GitStep::Push, None, GitFailureKind::Unknown, "");
        assert_eq!(nothing.message(), "Sync could not push: git gave no reason");

        let no_branch = GitFailure::local(GitStep::Sync, None, "no branch is checked out");
        assert_eq!(
            no_branch.message(),
            "Sync could not run: the failure was local \u{2014} no branch is checked out"
        );
    }

    /// The toast's one line names the step and keeps the cause, without the full report the modal
    /// and the log carry.
    #[test]
    fn toast_keeps_the_step_and_the_cause_line() {
        let failure = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "remote: Permission to o/r.git denied to someone.\n\
             fatal: unable to access 'https://github.com/o/r.git/': \
             The requested URL returned error: 403",
        );
        assert_eq!(
            failure.toast(),
            "Sync could not push: the remote refused access \u{2014} \
             remote: Permission to o/r.git denied to someone."
        );
        assert!(
            failure.toast().len() < failure.message().len(),
            "the toast is the short form: {}",
            failure.toast()
        );
    }

    /// The tooltip's single line keeps the claim and the line of git's text that carries the
    /// cause — never the `hint:`/advice lines git prints after it.
    #[test]
    fn short_keeps_the_cause_line() {
        let failure = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "remote: Permission to o/r.git denied to someone.\n\
             fatal: unable to access 'https://github.com/o/r.git/': \
             The requested URL returned error: 403",
        );
        assert_eq!(
            failure.short(),
            "the remote refused access \u{2014} remote: Permission to o/r.git denied to someone."
        );

        // An SSH refusal names the method first and git's generic summary after it.
        let ssh = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "git@github.com: Permission denied (publickey).\n\
             fatal: Could not read from remote repository.\n\n\
             Please make sure you have the correct access rights\n\
             and the repository exists.",
        );
        assert_eq!(
            ssh.short(),
            "the remote refused access \u{2014} git@github.com: Permission denied (publickey)."
        );

        // A rejected push names the ref and its reason before git's hints.
        let rejected = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::UpdateRejected,
            "To /tmp/remote.git\n\
             ! [rejected]        main -> main (non-fast-forward)\n\
             error: failed to push some refs to '/tmp/remote.git'\n\
             hint: Updates were rejected because the tip of your current branch is behind",
        );
        assert_eq!(
            rejected.short(),
            "the remote rejected the update \u{2014} \
             ! [rejected]        main -> main (non-fast-forward)"
        );

        // A long single line keeps both ends: git's cause can sit either way.
        let long = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::Unreachable,
            &format!(
                "fatal: unable to access '{}': The requested URL returned error: 403",
                "https://github.com/o/r/".repeat(12)
            ),
        );
        assert!(long.short().ends_with("error: 403"), "{}", long.short());
        assert!(
            long.short().len() < MAX_SHORT_REASON_BYTES + 80,
            "{}",
            long.short()
        );

        assert_eq!(
            GitFailure::new(GitStep::Push, None, GitFailureKind::Local, "").short(),
            "the failure was local"
        );
    }

    /// git reports its own measurements, which change between attempts at the same failure: the
    /// log dedupe key folds them out so one outage is one row.
    #[test]
    fn dedupe_key_folds_out_measurements() {
        let first = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::Unreachable,
            "fatal: unable to access 'https://x/': Failed to connect to x port 443 after 75002 ms: \
             Couldn't connect to server",
        );
        let second = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::Unreachable,
            "fatal: unable to access 'https://x/': Failed to connect to x port 443 after 75031 ms: \
             Couldn't connect to server",
        );
        assert_eq!(first.dedupe_key(), second.dedupe_key());

        // A different cause is a different key, even when only a status code differs.
        let forbidden = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: unable to access 'https://x/': The requested URL returned error: 403",
        );
        let unauthorized = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: unable to access 'https://x/': The requested URL returned error: 401",
        );
        assert_ne!(forbidden.dedupe_key(), unauthorized.dedupe_key());

        // The step is part of the key: a probe that failed the same way must not swallow the
        // pull or push that failed afterwards.
        let report = "fatal: unable to access 'https://x/': Could not resolve host: x";
        let probe = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::Unreachable,
            report,
        );
        let pull = GitFailure::new(
            GitStep::Pull,
            Some("origin"),
            GitFailureKind::Unreachable,
            report,
        );
        assert_ne!(probe.dedupe_key(), pull.dedupe_key());
    }

    /// Credential material is scrubbed once, at construction, so the toast, the tooltip, the
    /// modal and the never-pruned log row all carry the same safe text.
    #[test]
    fn credentials_are_scrubbed_from_every_surface() {
        let failure = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: unable to access 'https://oauth2:ghp_secret@github.com/o/r.git/': \
             The requested URL returned error: 403",
        );
        let message = failure.message();
        assert!(!message.contains("ghp_secret"), "{message}");
        assert!(!message.contains("oauth2"), "{message}");
        assert!(
            message.contains("https://***@github.com/o/r.git/"),
            "{message}"
        );

        // A password can carry an unencoded `/`: the colon form is matched across it, while a
        // path's `@` (no colon) is not mistaken for userinfo.
        let slash = GitFailure::new(
            GitStep::Push,
            None,
            GitFailureKind::AccessRefused,
            "fatal: unable to access 'https://user:pa/ss@github.com/o/r.git/': error: 403",
        )
        .message();
        assert!(!slash.contains("pa/ss"), "{slash}");
        assert!(slash.contains("https://***@github.com/o/r.git/"), "{slash}");

        let path = redact("fatal: bad ref 'https://github.com/o/r@v1.git'");
        assert!(path.contains("o/r@v1.git"), "{path}");
    }

    /// A pathological report is capped, and the cut keeps the tail: git's cause is in the last
    /// lines, so a head-only cut would lose it.
    #[test]
    fn cap_keeps_the_tail_with_the_cause() {
        let noise = "remote: counting objects\n".repeat(200);
        let failure = GitFailure::new(
            GitStep::Push,
            None,
            GitFailureKind::UpdateRejected,
            &format!("{noise}fatal: the repository is protected"),
        );
        assert!(
            failure.message().len() < MAX_REASON_BYTES + 200,
            "the stored text is capped: {}",
            failure.message().len()
        );
        assert!(
            failure
                .message()
                .ends_with("fatal: the repository is protected"),
            "{}",
            failure.message()
        );
        assert!(
            failure.message().contains("bytes omitted"),
            "{}",
            failure.message()
        );
    }

    /// The network layer's own failures are classified without reading git's text: no spawn is
    /// local, a deadline is no answer, and a silent exit states the exit instead of inventing a
    /// cause.
    #[test]
    fn network_failures_are_classified_by_origin() {
        let spawn = GitNetworkFailure::Spawn {
            command: "push".to_string(),
            error: "No such file or directory (os error 2)".to_string(),
        }
        .into_failure(GitStep::Push, Some("origin"));
        assert_eq!(spawn.kind(), GitFailureKind::Local);
        assert!(spawn.message().contains("could not be started"));

        let timed_out = GitNetworkFailure::TimedOut {
            command: "push".to_string(),
            secs: 300,
        }
        .into_failure(GitStep::Push, None);
        assert_eq!(timed_out.kind(), GitFailureKind::Unreachable);
        assert!(timed_out.message().contains("within 300s"));
        let wait_failed = GitNetworkFailure::WaitFailed {
            command: "push".to_string(),
            error: "No child processes (os error 10)".to_string(),
        }
        .into_failure(GitStep::Push, Some("origin"));
        assert_eq!(wait_failed.kind(), GitFailureKind::Local);
        assert!(
            wait_failed.message().contains("could not be waited for"),
            "{}",
            wait_failed.message()
        );

        let silent = GitNetworkFailure::Exited {
            command: "push".to_string(),
            status: Some(128),
            stderr: String::new(),
            stdout: String::new(),
        }
        .into_failure(GitStep::Push, None);
        assert_eq!(silent.kind(), GitFailureKind::Unknown);
        assert!(silent.message().contains("exited with status 128"));
    }
}
